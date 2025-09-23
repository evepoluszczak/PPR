import streamlit as st
import pandas as pd
from datetime import date, timedelta, datetime
import numpy as np

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Analyseur de PPR",
    page_icon="✈️",
    layout="wide"
)

# --- Fonctions de traitement des données (mises en cache pour la performance) ---

@st.cache_data
def load_and_prepare_data(uploaded_file):
    """Charge, lit et normalise les données du fichier CSV."""
    try:
        raw_df = pd.read_csv(uploaded_file, sep=';', encoding='latin-1')
        raw_df = raw_df.rename(columns={
            'CallSign': 'Call sign',
            'Registration': 'Immatriculation'
        })
        datetime_col = pd.to_datetime(raw_df['Date'], errors='coerce', dayfirst=True)
        raw_df['Slot.Date'] = datetime_col.dt.date
        raw_df['Slot.Hour'] = datetime_col.dt.time
        raw_df['Date / Heure Creation'] = datetime_col
        raw_df['Login (Suppression)'] = None
        raw_df.loc[raw_df['Deleted'] == True, 'Login (Suppression)'] = 'Deleted'
        
        if 'MovementTypeId' in raw_df.columns:
             movement_map = {True: 'Arrival', False: 'Departure'}
             raw_df['Type de mouvement'] = raw_df['MovementTypeId'].map(movement_map)
        return raw_df
    except Exception as e:
        st.error(f"Erreur lors de la lecture ou de la préparation du fichier : {e}")
        return None

@st.cache_data
def process_ppr_data(df):
    """Applique la logique de détection de doublons."""
    try:
        df_copy = df.copy()
        df_copy['Slot.Date'] = pd.to_datetime(df_copy['Slot.Date'], errors='coerce').dt.date
        df_copy.dropna(subset=['Slot.Date', 'Immatriculation'], inplace=True)
        active_ppr = df_copy[df_copy['Login (Suppression)'].isnull()].copy()

        group_cols = ['Slot.Date', 'Immatriculation']
        active_ppr['Nb de lignes'] = active_ppr.groupby(group_cols)['Slot.Date'].transform('count')
        duplicates = active_ppr[active_ppr['Nb de lignes'] > 1].copy()

        if 'Call sign' in duplicates.columns:
            duplicates = duplicates[duplicates['Call sign'] != 'RWYCHK']

        today = date.today()
        tomorrow = today + timedelta(days=1)
        duplicates = duplicates[duplicates['Slot.Date'].isin([today, tomorrow])]
        
        if duplicates.empty:
            return pd.DataFrame()

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
        final_result.drop(columns=['is_problematic_group'], inplace=True)
        return final_result
    except Exception as e:
        st.error(f"Une erreur est survenue lors du traitement des données : {e}")
        return pd.DataFrame()

# --- Fonctions d'affichage pour chaque page ---

def page_detection_doublons(df):
    """Affiche la page de détection des doublons."""
    st.title("✈️ Outil de détection et de suivi des PPR")
    st.markdown("Analyse des doublons et liste des vols pour **aujourd'hui** et **demain**.")
    
    # --- Calcul des KPIs et affichage ---
    st.header("📊 Tableau de bord")
    today = date.today()
    tomorrow = today + timedelta(days=1)
    
    active_ppr_full = df[df['Login (Suppression)'].isnull()].copy()
    active_ppr_full['Slot.Date'] = pd.to_datetime(active_ppr_full['Slot.Date']).dt.date
    
    ppr_today_count = active_ppr_full[active_ppr_full['Slot.Date'] == today].shape[0]
    ppr_tomorrow_count = active_ppr_full[active_ppr_full['Slot.Date'] == tomorrow].shape[0]
    
    result_df = process_ppr_data(df)
    
    num_groups = 0
    if not result_df.empty:
        num_groups = len(result_df.groupby(['Slot.Date', 'Immatriculation']))
        
    col1, col2, col3 = st.columns(3)
    col1.metric("PPR prévus aujourd'hui", ppr_today_count)
    col2.metric("PPR prévus demain", ppr_tomorrow_count)
    col3.metric("Groupes de doublons", num_groups, help="Nombre de groupes (Date, Immat.) avec des doublons problématiques.")

    # --- Section Analyse des Doublons ---
    st.header("🚨 Analyse des Doublons")
    if not result_df.empty:
        st.success(f"**{num_groups}** groupe(s) de doublons problématiques détecté(s) !")
        summary_df = result_df[result_df['Check'] != ''].copy()
        display_df = summary_df.rename(columns={'Slot.Date': 'Date du vol', 'Call sign': 'CallSign', 'Slot.Hour': 'Slot 1', 'Next_Slot.Hour': 'Slot 2', 'Type de mouvement': 'MovementType', 'OwnerProfileLogin': 'Login'})
        display_cols = ['Date du vol', 'Immatriculation', 'CallSign', 'Slot 1', 'Slot 2', 'MovementType', 'Login']
        display_cols_exist = [col for col in display_cols if col in display_df.columns]
        st.dataframe(display_df[display_cols_exist])
        
        st.download_button(label="📥 Télécharger les résultats complets de l'analyse en CSV", data=result_df.to_csv(index=False, sep=';').encode('utf-8'), file_name=f"ppr_doublons_details_{date.today()}.csv", mime="text/csv")
    else:
        st.success("🎉 Aucune PPR en doublon problématique détectée pour aujourd'hui et demain.")

    # --- Section Liste Complète ---
    st.header("📋 Liste des PPR Actifs (Aujourd'hui et Demain)")
    active_ppr_j0_j1 = active_ppr_full[active_ppr_full['Slot.Date'].isin([today, tomorrow])].copy()
    active_ppr_j0_j1.sort_values(by=['Slot.Date', 'Immatriculation', 'Slot.Hour'], inplace=True)
    with st.expander("Afficher/Masquer la liste complète des vols", expanded=False):
        filter_text = st.text_input("Filtrer la liste :", placeholder="Ex: HBLVK, T7-SCT, SFS...")
        display_cols = ['Slot.Date', 'Immatriculation', 'Call sign', 'Slot.Hour', 'Type de mouvement', 'HandlingAgentName', 'OwnerProfileLogin']
        display_cols_exist = [col for col in display_cols if col in active_ppr_j0_j1.columns]
        filtered_list = active_ppr_j0_j1
        if filter_text:
            mask = np.column_stack([filtered_list[col].astype(str).str.contains(filter_text, case=False, na=False) for col in display_cols_exist])
            filtered_list = filtered_list[mask.any(axis=1)]
        st.dataframe(filtered_list[display_cols_exist])

def page_analyse_visuelle(df):
    """Affiche la page d'analyse avec des graphiques."""
    st.title("📊 Analyse & Visualisations des PPR")

    today = date.today()
    tomorrow = today + timedelta(days=1)
    
    active_ppr = df[df['Login (Suppression)'].isnull()].copy()
    active_ppr['Slot.Date'] = pd.to_datetime(active_ppr['Slot.Date']).dt.date
    
    # Filtre de date pour les graphiques
    jour_choisi_str = st.selectbox(
        "Choisissez une journée à analyser",
        ("Aujourd'hui", "Demain")
    )
    
    jour_choisi = today if jour_choisi_str == "Aujourd'hui" else tomorrow
    
    st.header(f"Nombre de vols par heure pour le {jour_choisi.strftime('%d/%m/%Y')}")

    # Préparation des données pour le graphique
    df_jour = active_ppr[active_ppr['Slot.Date'] == jour_choisi].copy()

    if df_jour.empty:
        st.warning(f"Aucun vol prévu pour le {jour_choisi.strftime('%d/%m/%Y')}.")
    else:
        # Extraire l'heure de la colonne Slot.Hour
        df_jour['Heure'] = df_jour['Slot.Hour'].apply(lambda t: t.hour)

        # Compter les arrivées et départs par heure
        vols_par_heure = df_jour.groupby(['Heure', 'Type de mouvement']).size().unstack(fill_value=0)
        
        # S'assurer que toutes les heures de la journée sont présentes
        vols_par_heure = vols_par_heure.reindex(range(24), fill_value=0)
        
        # Renommer les colonnes pour la légende
        if 'Arrival' in vols_par_heure.columns:
            vols_par_heure.rename(columns={'Arrival': 'Arrivées'}, inplace=True)
        if 'Departure' in vols_par_heure.columns:
            vols_par_heure.rename(columns={'Departure': 'Départs'}, inplace=True)
            
        st.bar_chart(vols_par_heure)
        st.write("Ce graphique montre le nombre total de vols (PPR) prévus pour chaque heure de la journée sélectionnée, séparés par type de mouvement (arrivées et départs).")

# --- Interface principale de l'application ---

# Barre de navigation latérale
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choisissez une page", ["Détection Doublons", "Analyse & Visualisations"])

# Zone de chargement de fichier principale
uploaded_file = st.sidebar.file_uploader(
    "Choisissez le fichier CSV (`Reservations.csv`)",
    type=['csv']
)

if uploaded_file is not None:
    prepared_data = load_and_prepare_data(uploaded_file)
    
    if prepared_data is not None:
        if page == "Détection Doublons":
            page_detection_doublons(prepared_data)
        elif page == "Analyse & Visualisations":
            page_analyse_visuelle(prepared_data)
else:
    st.info("Veuillez charger un fichier CSV via la barre latérale pour commencer l'analyse.")

