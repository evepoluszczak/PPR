import streamlit as st
import pandas as pd
from datetime import date, timedelta
import numpy as np

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Détecteur de Doublons PPR",
    page_icon="✈️",
    layout="wide"
)

# Titre de l'application
st.title("✈️ Outil de détection et de suivi des PPR")
st.markdown("Chargez le fichier des réservations pour obtenir une vue d'ensemble des vols et identifier les doublons pour **aujourd'hui** et **demain**.")

def process_ppr_data(df):
    """
    Applique la logique de nettoyage et de détection de doublons sur un DataFrame standardisé.
    
    Args:
        df (pd.DataFrame): Le DataFrame normalisé contenant les données PPR.

    Returns:
        pd.DataFrame: Un DataFrame contenant uniquement les lignes de PPR en doublon,
                      ou un DataFrame vide si aucun doublon n'est trouvé.
    """
    try:
        # --- 1. Nettoyage et préparation ---
        
        df_copy = df.copy() # Travailler sur une copie pour éviter les effets de bord
        df_copy['Slot.Date'] = pd.to_datetime(df_copy['Slot.Date'], errors='coerce').dt.date
        df_copy['Date / Heure Creation'] = pd.to_datetime(df_copy['Date / Heure Creation'], errors='coerce')
        
        df_copy.dropna(subset=['Slot.Date', 'Call sign', 'Immatriculation'], inplace=True)

        active_ppr = df_copy[df_copy['Login (Suppression)'].isnull()].copy()

        # --- 2. Détection des doublons (Logique simple) ---

        group_cols = ['Slot.Date', 'Call sign', 'Immatriculation']
        
        active_ppr['Nb de lignes'] = active_ppr.groupby(group_cols)['Slot.Date'].transform('count')
        
        duplicates = active_ppr[active_ppr['Nb de lignes'] > 1].copy()

        duplicates = duplicates[duplicates['Call sign'] != 'RWYCHK']

        today = date.today()
        tomorrow = today + timedelta(days=1)
        duplicates = duplicates[duplicates['Slot.Date'].isin([today, tomorrow])]

        if 'approach speed Description' in duplicates.columns:
            duplicates = duplicates[duplicates['approach speed Description'] == 'case non cochée']
        
        if duplicates.empty:
            return pd.DataFrame()

        # --- 3. Analyse fine des doublons (Logique avancée) ---

        duplicates.sort_values(by=group_cols + ['Slot.Hour'], inplace=True)

        duplicates['Next_Slot.Hour'] = duplicates.groupby(group_cols)['Slot.Hour'].shift(-1)
        duplicates['Next_Type de mouvement'] = duplicates.groupby(group_cols)['Type de mouvement'].shift(-1)

        is_double = (duplicates['Type de mouvement'] == duplicates['Next_Type de mouvement']) & duplicates['Next_Type de mouvement'].notna()
        is_error = (duplicates['Slot.Hour'] == duplicates['Next_Slot.Hour']) & duplicates['Next_Slot.Hour'].notna()

        duplicates['Check'] = ''
        duplicates.loc[is_double, 'Check'] = 'Double'
        duplicates.loc[is_error, 'Check'] = 'Erreur'

        duplicates['is_problematic_group'] = duplicates.groupby(group_cols)['Check'].transform(lambda x: (x != '').any())

        final_result = duplicates[duplicates['is_problematic_group']].copy()

        final_result.drop(columns=[
            'Next_Slot.Hour', 
            'Next_Type de mouvement', 
            'is_problematic_group'
        ], inplace=True)
        
        return final_result

    except Exception as e:
        st.error(f"Une erreur est survenue lors du traitement des données : {e}")
        st.warning("Veuillez vérifier que le format du fichier correspond au modèle attendu.")
        return pd.DataFrame()


# --- Interface utilisateur ---

uploaded_file = st.file_uploader(
    "1. Choisissez le fichier CSV (`Reservations.csv`)",
    type=['csv']
)

if uploaded_file is not None:
    st.info(f"Fichier chargé : `{uploaded_file.name}`. Traitement en cours...")
    
    raw_df = None
    try:
        raw_df = pd.read_csv(uploaded_file, sep=';', encoding='latin-1')
        raw_df = raw_df.rename(columns={
            'CallSign': 'Call sign',
            'Registration': 'Immatriculation'
        })
        datetime_col = pd.to_datetime(raw_df['Date'], errors='coerce')
        raw_df['Slot.Date'] = datetime_col.dt.date
        raw_df['Slot.Hour'] = datetime_col.dt.time
        raw_df['Date / Heure Creation'] = datetime_col
        raw_df['Login (Suppression)'] = None
        raw_df.loc[raw_df['Deleted'] == True, 'Login (Suppression)'] = 'Deleted'
        
        movement_map = {'A': 'Arrival', 'D': 'Departure'}
        if 'MovementTypeId' in raw_df.columns:
             raw_df['Type de mouvement'] = raw_df['MovementTypeId'].map(movement_map)

    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier : {e}")
        raw_df = None
    
    if raw_df is not None:
        # --- Calcul des KPIs et affichage ---
        st.header("📊 Tableau de bord")

        today = date.today()
        tomorrow = today + timedelta(days=1)
        
        active_ppr_full = raw_df[raw_df['Login (Suppression)'].isnull()].copy()
        active_ppr_full['Slot.Date'] = pd.to_datetime(active_ppr_full['Slot.Date']).dt.date
        
        ppr_today_count = active_ppr_full[active_ppr_full['Slot.Date'] == today].shape[0]
        ppr_tomorrow_count = active_ppr_full[active_ppr_full['Slot.Date'] == tomorrow].shape[0]
        
        result_df = process_ppr_data(raw_df)
        
        num_groups = 0
        if not result_df.empty:
            num_groups = len(result_df.groupby(['Slot.Date', 'Call sign', 'Immatriculation']))
            
        col1, col2, col3 = st.columns(3)
        col1.metric("PPR prévus aujourd'hui", ppr_today_count)
        col2.metric("PPR prévus demain", ppr_tomorrow_count)
        col3.metric("Groupes de doublons", num_groups, help="Nombre de groupes (Date, Call sign, Immat.) avec des doublons problématiques.")

        # --- Section Analyse des Doublons ---
        st.header("🚨 Analyse des Doublons")

        if not result_df.empty:
            st.success(f"**{num_groups}** groupe(s) de doublons problématiques détecté(s) !")
            st.dataframe(result_df)
            
            st.write("### Actions recommandées :")
            st.write("- **Examinez** les groupes où une ligne est marquée `Double` ou `Erreur` dans la colonne `Check`.")
            st.write("- La marque indique que cette ligne et la **suivante** forment une paire problématique.")
            st.write("- **Contactez** les utilisateurs concernés pour régulariser la situation.")

            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(index=False, sep=';').encode('utf-8')

            csv = convert_df_to_csv(result_df)

            st.download_button(
               label="📥 Télécharger les résultats de l'analyse en CSV",
               data=csv,
               file_name=f"ppr_doublons_{date.today()}.csv",
               mime="text/csv",
            )
        else:
            st.success("🎉 Aucune PPR en doublon problématique détectée pour aujourd'hui et demain.")

        # --- Section Liste Complète ---
        st.header("📋 Liste des PPR Actifs (Aujourd'hui et Demain)")
        
        active_ppr_j0_j1 = active_ppr_full[active_ppr_full['Slot.Date'].isin([today, tomorrow])].copy()
        active_ppr_j0_j1.sort_values(by=['Slot.Date', 'Immatriculation', 'Slot.Hour'], inplace=True)
        
        with st.expander("Afficher/Masquer la liste complète des vols", expanded=True):
            
            filter_text = st.text_input("Filtrer la liste (par immatriculation, call sign, agent, etc.) :", placeholder="Ex: HBLVK, T7-SCT, SFS...")

            display_cols = ['Slot.Date', 'Immatriculation', 'Call sign', 'Slot.Hour', 'Type de mouvement', 'HandlingAgentName', 'OwnerProfileLogin']
            display_cols_exist = [col for col in display_cols if col in active_ppr_j0_j1.columns]
            
            filtered_list = active_ppr_j0_j1
            if filter_text:
                # Créer un masque booléen pour le filtrage sur plusieurs colonnes
                mask = np.column_stack([
                    filtered_list[col].astype(str).str.contains(filter_text, case=False, na=False) 
                    for col in display_cols_exist
                ])
                filtered_list = filtered_list[mask.any(axis=1)]

            if filtered_list.empty:
                st.warning("Aucun vol ne correspond à votre recherche.")
            else:
                st.dataframe(filtered_list[display_cols_exist])

else:
    st.info("En attente du chargement d'un fichier.")

