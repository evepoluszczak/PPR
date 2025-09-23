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
st.title("✈️ Outil de détection des PPR en doublon")
st.markdown("Cette application identifie les réservations PPR (Prior Permission Required) qui semblent être des doublons pour **aujourd'hui** et **demain**.")

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
        
        df['Slot.Date'] = pd.to_datetime(df['Slot.Date'], errors='coerce').dt.date
        df['Date / Heure Creation'] = pd.to_datetime(df['Date / Heure Creation'], errors='coerce')
        
        df.dropna(subset=['Slot.Date', 'Call sign', 'Immatriculation'], inplace=True)

        active_ppr = df[df['Login (Suppression)'].isnull()].copy()

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

        # Trier les groupes par heure pour comparer les vols consécutifs
        duplicates.sort_values(by=group_cols + ['Slot.Hour'], inplace=True)

        # Créer des colonnes décalées pour comparer une ligne avec la suivante dans le même groupe
        duplicates['Next_Slot.Hour'] = duplicates.groupby(group_cols)['Slot.Hour'].shift(-1)
        duplicates['Next_Type de mouvement'] = duplicates.groupby(group_cols)['Type de mouvement'].shift(-1)

        # Identifier les paires problématiques
        # Une paire est "Double" si les types de mouvement sont identiques
        is_double = (duplicates['Type de mouvement'] == duplicates['Next_Type de mouvement']) & duplicates['Next_Type de mouvement'].notna()
        # Une paire est "Erreur" si les heures sont identiques
        is_error = (duplicates['Slot.Hour'] == duplicates['Next_Slot.Hour']) & duplicates['Next_Slot.Hour'].notna()

        # Créer une colonne 'Check' pour marquer la première ligne de chaque paire problématique
        duplicates['Check'] = ''
        duplicates.loc[is_double, 'Check'] = 'Double'
        duplicates.loc[is_error, 'Check'] = 'Erreur'

        # Un groupe est problématique s'il contient au moins une paire problématique
        duplicates['is_problematic_group'] = duplicates.groupby(group_cols)['Check'].transform(lambda x: (x != '').any())

        # Ne garder que les groupes complets qui ont été identifiés comme problématiques
        final_result = duplicates[duplicates['is_problematic_group']].copy()

        # Nettoyer les colonnes temporaires avant l'affichage
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
    "1. Choisissez le fichier Excel (`PBI_PPR_EPL.xlsx`) ou CSV (`Reservations.csv`)",
    type=['xlsx', 'xls', 'csv']
)

if uploaded_file is not None:
    st.info(f"Fichier chargé : `{uploaded_file.name}`. Traitement en cours...")
    
    raw_df = None
    try:
        if uploaded_file.name.lower().endswith('.csv'):
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
            
            # Ajout de la colonne 'Type de mouvement' pour les fichiers CSV
            movement_map = {'A': 'Arrival', 'D': 'Departure'}
            if 'MovementTypeId' in raw_df.columns:
                 raw_df['Type de mouvement'] = raw_df['MovementTypeId'].map(movement_map)

        elif uploaded_file.name.lower().endswith(('.xls', '.xlsx')):
            excel_df = pd.read_excel(uploaded_file, sheet_name="PBI_PPR_J0_J5", header=None)
            excel_df.columns = excel_df.iloc[0]
            raw_df = excel_df.drop(excel_df.index[0]).reset_index(drop=True)
            raw_df = raw_df.rename(columns={
                "Slot Réservation.Date": "Slot.Date",
                "Heure": "Slot.Hour"
            })

    except Exception as e:
        if "Worksheet named 'PBI_PPR_J0_J5' not found" in str(e):
              st.error("Erreur : La feuille nommée `PBI_PPR_J0_J5` n'a pas été trouvée. Veuillez vérifier le fichier Excel.")
        else:
            st.error(f"Erreur lors de la lecture du fichier : {e}")
        raw_df = None
    
    if raw_df is not None:
        result_df = process_ppr_data(raw_df)

        st.header("2. Résultats de l'analyse")

        if not result_df.empty:
            group_cols = ['Slot.Date', 'Call sign', 'Immatriculation']
            num_groups = len(result_df.groupby(group_cols))

            st.success(f"**{num_groups}** groupe(s) de doublons problématiques détecté(s) !")
            
            st.dataframe(result_df)
            
            st.markdown("---")
            st.write("### Actions recommandées :")
            st.write("- **Examinez** les groupes où une ligne est marquée `Double` ou `Erreur` dans la colonne `Check`.")
            st.write("- La marque indique que cette ligne et la **suivante** forment une paire problématique.")
            st.write("- **Contactez** les utilisateurs concernés pour régulariser la situation.")

            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(index=False, sep=';').encode('utf-8')

            csv = convert_df_to_csv(result_df)

            st.download_button(
               label="📥 Télécharger les résultats en CSV",
               data=csv,
               file_name=f"ppr_doublons_{date.today()}.csv",
               mime="text/csv",
            )
        else:
            st.success("🎉 Aucune PPR en doublon problématique détectée pour aujourd'hui et demain.")
else:
    st.info("En attente du chargement d'un fichier.")

