import streamlit as st
import pandas as pd
from datetime import date, timedelta, datetime
import numpy as np
import calendar
import altair as alt

# --- Configuration de la page Streamlit ---
st.set_page_config(
    page_title="Tableau de Bord Opérationnel",
    page_icon="✈️",
    layout="wide"
)

# --- Initialisation de l'état de session ---
# Fichiers de données
if 'ppr_detail_data' not in st.session_state: st.session_state.ppr_detail_data = None
if 'saturation_data' not in st.session_state: st.session_state.saturation_data = None
if 'predicted_data' not in st.session_state: st.session_state.predicted_data = None
if 'actual_data' not in st.session_state: st.session_state.actual_data = None
# Données calculées
if 'anomalies_df' not in st.session_state: st.session_state.anomalies_df = pd.DataFrame()


# --- Fichier de données statique ---
CAPACITIES_FILE = 'capacities.xlsx'

# --- Fonctions de traitement des données (mises en cache pour la performance) ---

@st.cache_data
def load_and_prepare_data(uploaded_file, file_type):
    """Charge, lit et normalise les données des différents types de fichiers."""
    try:
        if file_type == 'PPR_DETAIL':
            raw_df = pd.read_csv(uploaded_file, sep=';', encoding='latin-1')
            required_cols = ['Id', 'Date', 'CallSign', 'Registration', 'MovementTypeId', 'Deleted', 'HandlingAgentName', 'EngineDescription']
            if not all(col in raw_df.columns for col in required_cols):
                st.error("Le fichier PPR détaillé semble invalide.")
                return None
            raw_df = raw_df.rename(columns={'CallSign': 'Call sign', 'Registration': 'Immatriculation'})
            datetime_col = pd.to_datetime(raw_df['Date'], errors='coerce', dayfirst=True)
            raw_df['Slot.Date'] = datetime_col.dt.date
            raw_df['Slot.Hour'] = datetime_col.dt.time
            raw_df['Date / Heure Creation'] = datetime_col
            raw_df['Login (Suppression)'] = None
            raw_df.loc[raw_df['Deleted'] == True, 'Login (Suppression)'] = 'Deleted'
            movement_map = {True: 'Arrival', False: 'Departure'}
            raw_df['Type de mouvement'] = raw_df['MovementTypeId'].map(movement_map)
            return raw_df

        elif file_type == 'COMBINED':
            raw_df = pd.read_excel(uploaded_file)
            required_cols = ['Date', 'Heure_Local_Tab', 'Rotation', 'Nombre de réservations']
            if not all(col in raw_df.columns for col in required_cols):
                st.error("Le fichier combiné (PPR+SCR) semble invalide.")
                return None
            raw_df['Slot.Date'] = pd.to_datetime(raw_df['Date'], errors='coerce').dt.date
            raw_df['Heure'] = raw_df['Heure_Local_Tab'].str.split('h').str[0].astype(int)
            raw_df.rename(columns={'Rotation': 'Vols SCR', 'Nombre de réservations': 'Vols PPR'}, inplace=True)
            raw_df['Vols PPR'] = raw_df['Vols PPR'].fillna(0).astype(int)
            return raw_df
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier {file_type}: {e}")
        return None

@st.cache_data
def process_ppr_data(df):
    """Applique la logique de détection de doublons."""
    if df is None: return pd.DataFrame()
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
        if duplicates.empty: return pd.DataFrame()
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
    except Exception: return pd.DataFrame()

# --- Pages de l'application ---

def page_accueil():
    st.title("🏠 Accueil & Tableau de Bord Principal")
    st.markdown(f"Bienvenue sur votre tableau de bord opérationnel. Nous sommes le **{date.today().strftime('%A %d %B %Y')}**.")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Anomalies PPR du jour")
        if st.session_state.ppr_detail_data is not None:
            anomalies_df = process_ppr_data(st.session_state.ppr_detail_data)
            st.session_state.anomalies_df = anomalies_df # Save for other pages
            num_anomalies = anomalies_df[anomalies_df['Check'] != ''].shape[0] if not anomalies_df.empty else 0
            st.metric("Anomalies détectées", num_anomalies)
        else:
            st.info("Chargez le fichier PPR Détaillé pour voir les anomalies.")

    with col2:
        st.subheader("Heure la plus chargée (J0)")
        if st.session_state.saturation_data is not None:
            today_str = date.today().strftime('%Y-%m-%d')
            today_data = st.session_state.saturation_data[st.session_state.saturation_data['Date'] == today_str]
            if not today_data.empty:
                hourly_load = today_data.groupby('Heure')[['Vols PPR', 'Vols SCR']].sum()
                hourly_load['Total'] = hourly_load['Vols PPR'] + hourly_load['Vols SCR']
                peak_hour = hourly_load['Total'].idxmax()
                peak_load = hourly_load['Total'].max()
                st.metric(f"Heure de pointe : {peak_hour}:00", f"{peak_load} vols")
            else:
                st.info("Aucune donnée de saturation pour aujourd'hui.")
        else:
            st.info("Chargez le fichier de prévisions pour voir l'heure de pointe.")

    with col3:
        st.subheader("Delta Prévu/Réel (J-1)")
        if st.session_state.predicted_data is not None and st.session_state.actual_data is not None:
            yesterday = date.today() - timedelta(days=1)
            predicted_counts = calculate_hourly_counts(st.session_state.predicted_data, yesterday)
            actual_counts = calculate_hourly_counts(st.session_state.actual_data, yesterday)
            total_predicted = predicted_counts['Total Vols_Prévu'].sum()
            total_actual = actual_counts['Total Vols_Réalisé'].sum()
            delta = total_actual - total_predicted
            st.metric("Delta total de la veille", f"{delta:+} vols")
        else:
            st.info("Chargez les fichiers Prévu et Réalisé pour voir le delta.")
            
def page_detection_doublons():
    st.title("🚨 Détection des Doublons PPR")
    if st.session_state.ppr_detail_data is None:
        st.info("Veuillez charger le fichier PPR Détaillé via la barre latérale pour commencer.")
        return

    df = st.session_state.ppr_detail_data
    result_df = st.session_state.anomalies_df
    summary_df = result_df[result_df['Check'] != ''].copy() if not result_df.empty else pd.DataFrame()
    num_anomalies = len(summary_df)

    if num_anomalies > 0:
        st.success(f"**{num_anomalies}** anomalie(s) détectée(s) !")
        
        # Filtre par login
        logins = summary_df['OwnerProfileLogin'].dropna().unique()
        selected_login = st.selectbox("Filtrer par login", ["Tous"] + list(logins))
        
        if selected_login != "Tous":
            summary_df = summary_df[summary_df['OwnerProfileLogin'] == selected_login]

        display_df = summary_df.rename(columns={'Slot.Date': 'Date du vol', 'Call sign': 'CallSign', 'Slot.Hour': 'Slot 1', 'Next_Slot.Hour': 'Slot 2', 'Type de mouvement': 'MovementType', 'OwnerProfileLogin': 'Login'})
        display_cols = ['Date du vol', 'Immatriculation', 'CallSign', 'Slot 1', 'Slot 2', 'MovementType', 'Login']
        display_cols_exist = [col for col in display_cols if col in display_df.columns]
        
        def highlight_same_slot(row):
            return ['background-color: #ffcccc'] * len(row.index) if pd.notna(row['Slot 1']) and pd.notna(row['Slot 2']) and row['Slot 1'] == row['Slot 2'] else [''] * len(row.index)
        
        st.dataframe(display_df[display_cols_exist].style.apply(highlight_same_slot, axis=1))
        # ... (le reste de la page reste similaire : génération de mail, liste complète, etc.)
    else:
        st.success("🎉 Aucune anomalie détectée.")

def page_saturation_piste():
    st.title("🚦 Analyse de Saturation Piste")
    if st.session_state.saturation_data is None:
        st.info("Veuillez charger le fichier de prévisions combiné (PPR+SCR) via la barre latérale.")
        return
    
    combined_df = st.session_state.saturation_data
    available_dates = sorted(combined_df['Slot.Date'].unique())
    
    if not available_dates:
        st.warning("Le fichier chargé ne contient aucune date valide.")
        return

    jour_choisi = st.selectbox("Choisissez une journée à analyser", available_dates, format_func=lambda d: d.strftime('%d/%m/%Y'))
    st.header(f"Analyse pour le {jour_choisi.strftime('%d/%m/%Y')}")

    try:
        capacities_df = pd.read_excel(CAPACITIES_FILE)
    except FileNotFoundError:
        st.error(f"Fichier de capacités '{CAPACITIES_FILE}' non trouvé.")
        return

    season = get_season(jour_choisi)
    day_name = jour_choisi.strftime('%A')
    
    capacity_day_df = capacities_df[(capacities_df['Saison'] == season) & (capacities_df['JourSemaine'] == day_name)]
    if capacity_day_df.empty:
        st.error(f"Impossible de trouver les capacités pour {season} / {day_name}.")
        return
    capacity_day_df = capacity_day_df.set_index('Heure')[['Capacité Totale', 'Capacité Arrivées']]
    
    # ... (le reste de la logique de la page est conservé, mais utilise la variable `jour_choisi`)

def page_post_operationnelle():
    st.title("🔎 Analyse Post-Opérationnelle")
    if st.session_state.predicted_data is None or st.session_state.actual_data is None:
        st.info("Veuillez charger les fichiers Prévu et Réalisé via la barre latérale.")
        return
        
    # ... (le reste de la logique de la page est conservé)

def page_analyse_agent():
    st.title("👨‍✈️ Analyse par Agent de Handling")
    if st.session_state.ppr_detail_data is None:
        st.info("Veuillez charger le fichier PPR Détaillé via la barre latérale.")
        return
    
    df = st.session_state.ppr_detail_data[st.session_state.ppr_detail_data['Login (Suppression)'].isnull()].copy()
    df.dropna(subset=['HandlingAgentName'], inplace=True)
    
    st.header("Nombre de vols par Agent")
    agent_counts = df['HandlingAgentName'].value_counts()
    st.bar_chart(agent_counts)
    
    st.header("Anomalies par Agent")
    anomalies_df = st.session_state.anomalies_df
    if not anomalies_df.empty:
        anomalies_by_agent = anomalies_df.groupby('HandlingAgentName').size().rename("Nombre d'anomalies")
        st.dataframe(anomalies_by_agent)
    else:
        st.success("Aucune anomalie détectée pour calculer les statistiques par agent.")

def page_analyse_aeronef():
    st.title("✈️ Analyse par Type d'Aéronef")
    if st.session_state.ppr_detail_data is None:
        st.info("Veuillez charger le fichier PPR Détaillé via la barre latérale.")
        return
        
    df = st.session_state.ppr_detail_data[st.session_state.ppr_detail_data['Login (Suppression)'].isnull()].copy()
    df.dropna(subset=['EngineDescription'], inplace=True)

    st.header("Top 15 des types d'aéronefs les plus fréquents")
    aircraft_counts = df['EngineDescription'].value_counts().nlargest(15)
    st.bar_chart(aircraft_counts)

    st.header("Répartition horaire par type d'aéronef")
    df['Heure'] = df['Slot.Hour'].apply(lambda t: t.hour)
    hourly_aircraft = df.groupby(['Heure', 'EngineDescription']).size().unstack(fill_value=0)
    top_5_aircraft = df['EngineDescription'].value_counts().nlargest(5).index
    st.area_chart(hourly_aircraft[top_5_aircraft])


# --- Fonctions utilitaires (get_season, calculate_hourly_counts) ---
# ... (ces fonctions restent les mêmes)
def get_season(dt):
    year = dt.year
    mar_last_day = calendar.monthrange(year, 3)[1]
    mar_last_sun = mar_last_day - (datetime(year, 3, mar_last_day).weekday() + 1) % 7
    summer_start = datetime(year, 3, mar_last_sun)
    oct_last_day = calendar.monthrange(year, 10)[1]
    oct_last_sun = oct_last_day - (datetime(year, 10, oct_last_day).weekday() + 1) % 7
    winter_start = datetime(year, 10, oct_last_sun)
    return "Summer" if summer_start <= datetime(year, dt.month, dt.day) < winter_start else "Winter"

def calculate_hourly_counts(df, analysis_day):
    df_jour = df[df['Slot.Date'] == analysis_day].copy()
    # ... (le reste de la fonction est inchangé)


# --- Interface principale de l'application ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choisissez une page", ["Accueil", "Détection Doublons", "Analyse de Saturation Piste", "Analyse Post-Opérationnelle", "Analyse par Agent", "Analyse par Aéronef"])

st.sidebar.title("Fichiers de données")

# Section de chargement centralisée
ppr_detail_file = st.sidebar.file_uploader("1. PPR Détaillé (`Reservations.csv`)", type=['csv'])
saturation_file = st.sidebar.file_uploader("2. Prévisions Combiné (PPR+SCR)", type=['xlsx', 'xls'])
st.sidebar.markdown("---")
st.sidebar.markdown("**Pour l'analyse post-opérationnelle :**")
predicted_file = st.sidebar.file_uploader("3. Fichier Prévu (J-1)", type=['xlsx', 'xls'], key="predicted")
actual_file = st.sidebar.file_uploader("4. Fichier Réalisé (J0)", type=['xlsx', 'xls'], key="actual")

# Mise à jour de l'état de session
if ppr_detail_file: st.session_state.ppr_detail_data = load_and_prepare_data(ppr_detail_file, 'PPR_DETAIL')
if saturation_file: st.session_state.saturation_data = load_and_prepare_data(saturation_file, 'COMBINED')
if predicted_file: st.session_state.predicted_data = load_and_prepare_data(predicted_file, 'COMBINED')
if actual_file: st.session_state.actual_data = load_and_prepare_data(actual_file, 'COMBINED')


# Affichage de la page sélectionnée
if page == "Accueil":
    page_accueil()
elif page == "Détection Doublons":
    page_detection_doublons()
elif page == "Analyse de Saturation Piste":
    page_saturation_piste()
elif page == "Analyse Post-Opérationnelle":
    page_post_operationnelle()
elif page == "Analyse par Agent":
    page_analyse_agent()
elif page == "Analyse par Aéronef":
    page_analyse_aeronef()

