import streamlit as st
import pandas as pd
from datetime import date, timedelta, datetime
import numpy as np
import calendar
import altair as alt

# --- Configuration de la page Streamlit ---
st.set_page_config(
    page_title="Analyseur de PPR & Piste",
    page_icon="✈️",
    layout="wide"
)

# --- Fichiers de données statiques ---
CAPACITIES_FILE = 'capacities.xlsx'

# --- Fonctions de traitement des données (mises en cache pour la performance) ---

@st.cache_data
def load_and_prepare_data(uploaded_file, file_type):
    """Charge, lit et normalise les données du fichier (PPR en CSV, SCR en XLSX)."""
    try:
        if file_type == 'PPR':
            raw_df = pd.read_csv(uploaded_file, sep=';', encoding='latin-1')
            required_cols = ['Id', 'Date', 'CallSign', 'Registration', 'MovementTypeId', 'Deleted']
            if not all(col in raw_df.columns for col in required_cols):
                st.error(f"Le fichier PPR semble invalide. Colonnes attendues: {', '.join(required_cols)}.")
                return None

            raw_df = raw_df.rename(columns={'CallSign': 'Call sign', 'Registration': 'Immatriculation'})
            datetime_col = pd.to_datetime(raw_df['Date'], errors='coerce', dayfirst=True)
            raw_df['Slot.Date'] = datetime_col.dt.date
            raw_df['Slot.Hour'] = datetime_col.dt.time
            raw_df['Date / Heure Creation'] = datetime_col
            raw_df['Login (Suppression)'] = None
            raw_df.loc[raw_df['Deleted'] == True, 'Login (Suppression)'] = 'Deleted'
            if 'MovementTypeId' in raw_df.columns:
                 movement_map = {True: 'Arrival', False: 'Departure'}
                 raw_df['Type de mouvement'] = raw_df['MovementTypeId'].map(movement_map)
        
        elif file_type == 'SCR':
            raw_df = pd.read_excel(uploaded_file) # Lire le fichier Excel
            required_cols = ['Date', 'Heure_Local_Tab', 'Rotation']
            if not all(col in raw_df.columns for col in required_cols):
                st.error(f"Le fichier SCR semble invalide. Colonnes attendues: {', '.join(required_cols)}.")
                return None
            
            raw_df['Slot.Date'] = pd.to_datetime(raw_df['Date'], errors='coerce').dt.date
            raw_df['Heure'] = raw_df['Heure_Local_Tab'].str.split('h').str[0].astype(int)

        return raw_df
    except Exception as e:
        st.error(f"Erreur lors de la lecture ou de la préparation du fichier {file_type}: {e}")
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

# --- Fonctions d'affichage pour chaque page ---

def page_detection_doublons(df):
    """Affiche la page de détection des doublons."""
    st.title("✈️ Outil de détection et de suivi des PPR")
    st.markdown("Analyse des doublons et liste des vols pour **aujourd'hui** et **demain**.")
    st.header("📊 Tableau de bord")
    today = date.today()
    tomorrow = today + timedelta(days=1)
    active_ppr_full = df[df['Login (Suppression)'].isnull()].copy()
    active_ppr_full['Slot.Date'] = pd.to_datetime(active_ppr_full['Slot.Date']).dt.date
    ppr_today_count = active_ppr_full[active_ppr_full['Slot.Date'] == today].shape[0]
    ppr_tomorrow_count = active_ppr_full[active_ppr_full['Slot.Date'] == tomorrow].shape[0]
    result_df = process_ppr_data(df)
    num_anomalies = 0
    if not result_df.empty:
        num_anomalies = result_df[result_df['Check'] != ''].shape[0]
    col1, col2, col3 = st.columns(3)
    col1.metric("PPR prévus aujourd'hui", ppr_today_count)
    col2.metric("PPR prévus demain", ppr_tomorrow_count)
    col3.metric("Anomalies détectées", num_anomalies, help="Nombre de paires de vols problématiques.")
    st.header("🚨 Analyse des Doublons")
    summary_df = pd.DataFrame()
    if num_anomalies > 0:
        st.success(f"**{num_anomalies}** anomalie(s) détectée(s) !")
        summary_df = result_df[result_df['Check'] != ''].copy()
        display_df = summary_df.rename(columns={'Slot.Date': 'Date du vol', 'Call sign': 'CallSign', 'Slot.Hour': 'Slot 1', 'Next_Slot.Hour': 'Slot 2', 'Type de mouvement': 'MovementType', 'OwnerProfileLogin': 'Login'})
        display_cols = ['Date du vol', 'Immatriculation', 'CallSign', 'Slot 1', 'Slot 2', 'MovementType', 'Login']
        display_cols_exist = [col for col in display_cols if col in display_df.columns]
        def highlight_same_slot(row):
            if pd.notna(row['Slot 1']) and pd.notna(row['Slot 2']) and row['Slot 1'] == row['Slot 2']:
                return ['background-color: #ffcccc'] * len(row.index)
            else:
                return [''] * len(row.index)
        st.dataframe(display_df[display_cols_exist].style.apply(highlight_same_slot, axis=1))
        st.download_button(label="📥 Télécharger les résultats CSV", data=result_df.to_csv(index=False, sep=';').encode('utf-8'), file_name=f"ppr_doublons_details_{date.today()}.csv", mime="text/csv")
    else:
        st.success("🎉 Aucune anomalie détectée.")
    st.header("📧 Générer les mails de correction")
    if st.button("Générer le texte pour chaque utilisateur"):
        if num_anomalies > 0 and not summary_df.empty:
            logins_to_notify = summary_df['OwnerProfileLogin'].dropna().unique()
            if len(logins_to_notify) > 0:
                for login in logins_to_notify:
                    with st.expander(f"Mail pour {login}"):
                        user_anomalies = summary_df[summary_df['OwnerProfileLogin'] == login]
                        mail_body_lines = ["Bonjour,", "\nNos systèmes ont détecté des anomalies dans vos réservations PPR. Pourriez-vous les corriger ?\n"]
                        for index, row in user_anomalies.iterrows():
                            reason = "Horaires identiques" if row['Check'] == 'Erreur' else f"Deux '{row['Type de mouvement']}' consécutifs"
                            line = f"- Immat: {row['Immatriculation']}, CallSign: {row.get('Call sign', 'N/A')}, Slots: {row['Slot.Hour']} & {row['Next_Slot.Hour']} -> Motif: {reason}"
                            mail_body_lines.append(line)
                        mail_body_lines.extend(["\nMerci de votre collaboration.", "Cordialement,"])
                        full_mail_text = "\n".join(mail_body_lines)
                        st.text_area("Texte à copier :", full_mail_text, height=250, key=f"mail_{login.replace('.', '_')}")
            else:
                st.write("Aucun login associé aux anomalies.")
        else:
            st.info("Aucune anomalie à signaler.")
    st.header("📋 Liste des PPR Actifs")
    active_ppr_j0_j1 = active_ppr_full[active_ppr_full['Slot.Date'].isin([today, tomorrow])].copy()
    active_ppr_j0_j1.sort_values(by=['Slot.Date', 'Immatriculation', 'Slot.Hour'], inplace=True)
    with st.expander("Afficher/Masquer la liste complète", expanded=False):
        filter_text = st.text_input("Filtrer la liste :", placeholder="Ex: HBLVK, T7-SCT, SFS...")
        display_cols = ['Slot.Date', 'Immatriculation', 'Call sign', 'Slot.Hour', 'Type de mouvement', 'HandlingAgentName', 'OwnerProfileLogin']
        display_cols_exist = [col for col in display_cols if col in active_ppr_j0_j1.columns]
        filtered_list = active_ppr_j0_j1
        if filter_text:
            mask = np.column_stack([filtered_list[col].astype(str).str.contains(filter_text, case=False, na=False) for col in display_cols_exist])
            filtered_list = filtered_list[mask.any(axis=1)]
        st.dataframe(filtered_list[display_cols_exist])

def page_analyse_visuelle(df):
    """Affiche la page d'analyse PPR avec des graphiques."""
    st.title("📊 Analyse & Visualisations des PPR")
    today = date.today()
    tomorrow = today + timedelta(days=1)
    active_ppr = df[df['Login (Suppression)'].isnull()].copy()
    active_ppr['Slot.Date'] = pd.to_datetime(active_ppr['Slot.Date']).dt.date
    jour_choisi_str = st.selectbox("Choisissez une journée à analyser", ("Aujourd'hui", "Demain"))
    show_rwy_check = st.checkbox("Mettre en évidence les RWYCHK")
    jour_choisi = today if jour_choisi_str == "Aujourd'hui" else tomorrow
    st.header(f"Nombre de vols par heure pour le {jour_choisi.strftime('%d/%m/%Y')}")
    df_jour = active_ppr[active_ppr['Slot.Date'] == jour_choisi].copy()
    if df_jour.empty:
        st.warning(f"Aucun vol PPR prévu pour le {jour_choisi.strftime('%d/%m/%Y')}.")
    else:
        df_jour['Heure'] = df_jour['Slot.Hour'].apply(lambda t: t.hour)
        df_flights = df_jour[df_jour['Call sign'] != 'RWYCHK']
        vols_par_heure = df_flights.groupby(['Heure', 'Type de mouvement']).size().unstack(fill_value=0)
        vols_par_heure = vols_par_heure.reindex(range(24), fill_value=0)
        if 'Arrival' in vols_par_heure.columns: vols_par_heure.rename(columns={'Arrival': 'Arrivées'}, inplace=True)
        if 'Departure' in vols_par_heure.columns: vols_par_heure.rename(columns={'Departure': 'Départs'}, inplace=True)
        if show_rwy_check:
            df_rwy = df_jour[df_jour['Call sign'] == 'RWYCHK']
            if not df_rwy.empty:
                rwy_par_heure = df_rwy.groupby('Heure').size().rename('RWYCHK')
                vols_par_heure = pd.concat([vols_par_heure, rwy_par_heure], axis=1).fillna(0)
                vols_par_heure['RWYCHK'] = vols_par_heure['RWYCHK'].astype(int)
        st.bar_chart(vols_par_heure)

def get_season(dt):
    """Détermine la saison IATA (Winter/Summer) pour une date donnée."""
    year = dt.year
    mar_last_day = calendar.monthrange(year, 3)[1]
    mar_last_sun = mar_last_day - (datetime(year, 3, mar_last_day).weekday() + 1) % 7
    summer_start = datetime(year, 3, mar_last_sun)
    oct_last_day = calendar.monthrange(year, 10)[1]
    oct_last_sun = oct_last_day - (datetime(year, 10, oct_last_day).weekday() + 1) % 7
    winter_start = datetime(year, 10, oct_last_sun)
    if summer_start <= datetime(year, dt.month, dt.day) < winter_start:
        return "Summer"
    else:
        return "Winter"

def page_saturation_piste(ppr_df, scr_df):
    """Affiche la page d'analyse de la saturation de la piste."""
    st.title("🚦 Analyse de Saturation Piste")
    st.markdown("Compare la charge de vols (PPR + SCR) à la capacité théorique de la piste.")
    
    jour_choisi_str = st.selectbox("Choisissez une journée à analyser", ("Aujourd'hui", "Demain"), key="saturation_day")
    today = date.today()
    jour_choisi = today if jour_choisi_str == "Aujourd'hui" else today + timedelta(days=1)
    
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

    ppr_df['Slot.Date'] = pd.to_datetime(ppr_df['Slot.Date']).dt.date
    ppr_jour = ppr_df[ppr_df['Slot.Date'] == jour_choisi].copy()
    ppr_jour['Heure'] = ppr_jour['Slot.Hour'].apply(lambda t: t.hour)
    ppr_counts = ppr_jour.groupby('Heure').size().rename('Vols PPR')

    scr_df['Slot.Date'] = pd.to_datetime(scr_df['Slot.Date']).dt.date
    scr_jour = scr_df[scr_df['Slot.Date'] == jour_choisi].copy()
    if not scr_jour.empty:
        scr_counts = scr_jour.groupby('Heure')['Rotation'].sum().rename('Vols SCR')
    else:
        scr_counts = pd.Series(name='Vols SCR', dtype=int)

    analysis_df = pd.DataFrame(index=range(24))
    analysis_df = analysis_df.join(capacity_day_df)
    analysis_df = analysis_df.join(ppr_counts)
    analysis_df = analysis_df.join(scr_counts)
    analysis_df.fillna(0, inplace=True)
    analysis_df['Total Vols'] = analysis_df['Vols PPR'] + analysis_df['Vols SCR']
    analysis_df['Capacité Résiduelle'] = analysis_df['Capacité Totale'] - analysis_df['Total Vols']
    analysis_df = analysis_df.astype(int)

    st.subheader("Graphique de charge de la piste")
    
    # Préparation des données pour le graphique Altair
    source = analysis_df.reset_index().rename(columns={'index': 'Heure'})
    source_melted = source.melt(
        id_vars=['Heure', 'Capacité Totale'],
        value_vars=['Vols PPR', 'Vols SCR'],
        var_name='Type de Vol',
        value_name='Nombre de Vols'
    )

    # Création du graphique en barres empilées
    bars = alt.Chart(source_melted).mark_bar().encode(
        x=alt.X('Heure:O', title='Heure', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('sum(Nombre de Vols):Q', title='Nombre de Vols'),
        color=alt.Color('Type de Vol:N', title='Type de Vol', scale=alt.Scale(domain=['Vols PPR', 'Vols SCR'], range=['#1f77b4', '#ff7f0e'])),
        tooltip=['Heure', 'Type de Vol', 'sum(Nombre de Vols)']
    )

    # Création de la ligne de capacité
    line = alt.Chart(source).mark_line(color='red', strokeDash=[5,5], size=3).encode(
        x=alt.X('Heure:O'),
        y=alt.Y('Capacité Totale:Q'),
        tooltip=['Heure', 'Capacité Totale']
    )
    
    # Combinaison des deux graphiques
    chart = (bars + line).properties(
        title=f"Charge de la piste vs. Capacité ({jour_choisi.strftime('%d/%m/%Y')})"
    ).resolve_scale(
        y='shared'
    )
    
    st.altair_chart(chart, use_container_width=True)
    
    st.subheader("Détails par heure")
    st.dataframe(analysis_df)

# --- Interface principale de l'application ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choisissez une page", ["Détection Doublons", "Analyse & Visualisations", "Analyse de Saturation Piste"])

st.sidebar.title("Fichiers de données")
ppr_uploaded_file = st.sidebar.file_uploader("1. Fichier PPR (`Reservations.csv`)", type=['csv'])

scr_uploaded_file = None
if page == "Analyse de Saturation Piste":
    scr_uploaded_file = st.sidebar.file_uploader("2. Fichier SCR (Prévisions Skyguide)", type=['xlsx', 'xls'])

if page in ["Détection Doublons", "Analyse & Visualisations"]:
    if ppr_uploaded_file:
        ppr_data = load_and_prepare_data(ppr_uploaded_file, 'PPR')
        if ppr_data is not None:
            if page == "Détection Doublons": page_detection_doublons(ppr_data)
            elif page == "Analyse & Visualisations": page_analyse_visuelle(ppr_data)
    else:
        st.info("Veuillez charger un fichier PPR via la barre latérale. [Cliquez ici pour récupérer le fichier](https://ppr.gva.ch/Reservations/Index).")
elif page == "Analyse de Saturation Piste":
    if ppr_uploaded_file and scr_uploaded_file:
        ppr_data = load_and_prepare_data(ppr_uploaded_file, 'PPR')
        scr_data = load_and_prepare_data(scr_uploaded_file, 'SCR')
        if ppr_data is not None and scr_data is not None:
            page_saturation_piste(ppr_data, scr_data)
    else:
        st.info("Veuillez charger le fichier PPR et le fichier SCR via la barre latérale pour lancer l'analyse de saturation.")

