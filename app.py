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

# --- Initialisation de l'état de session ---
if 'ppr_data' not in st.session_state:
    st.session_state.ppr_data = None
if 'saturation_data' not in st.session_state:
    st.session_state.saturation_data = None
if 'predicted_data' not in st.session_state:
    st.session_state.predicted_data = None
if 'actual_data' not in st.session_state:
    st.session_state.actual_data = None

# --- Fichiers de données statiques ---
CAPACITIES_FILE = 'capacities.xlsx'

# --- Fonctions de traitement des données (mises en cache pour la performance) ---

@st.cache_data
def load_and_prepare_data(uploaded_file, file_type):
    """Charge, lit et normalise les données des différents types de fichiers."""
    try:
        if file_type == 'PPR_DETAIL': # Pour la détection de doublons
            raw_df = pd.read_csv(uploaded_file, sep=';', encoding='latin-1')
            required_cols = ['Id', 'Date', 'CallSign', 'Registration', 'MovementTypeId', 'Deleted']
            if not all(col in raw_df.columns for col in required_cols):
                st.error(f"Le fichier PPR détaillé semble invalide.")
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

        elif file_type == 'COMBINED': # Pour la saturation et l'analyse post-op
            raw_df = pd.read_excel(uploaded_file)
            required_cols = ['Date', 'Heure_Local_Tab', 'Rotation', 'Nombre de réservations']
            if not all(col in raw_df.columns for col in required_cols):
                st.error(f"Le fichier combiné (PPR+SCR) semble invalide.")
                return None
            raw_df['Slot.Date'] = pd.to_datetime(raw_df['Date'], errors='coerce').dt.date
            raw_df['Heure'] = raw_df['Heure_Local_Tab'].str.split('h').str[0].astype(int)
            # Renommer pour clarté
            raw_df.rename(columns={'Rotation': 'Vols SCR', 'Nombre de réservations': 'Vols PPR'}, inplace=True)
            raw_df['Vols PPR'] = raw_df['Vols PPR'].fillna(0).astype(int) # Gérer les cas où il n'y a pas de PPR
            return raw_df
            
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier {file_type}: {e}")
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
                            # Traduction du type de mouvement
                            movement_translation = {'Arrival': 'Arrivée', 'Departure': 'Départ'}
                            translated_movement = movement_translation.get(row['Type de mouvement'], row['Type de mouvement'])
                            
                            # Motif de l'anomalie
                            reason = "Horaires identiques" if row['Check'] == 'Erreur' else f"Deux '{translated_movement}' consécutifs"
                            
                            # Formatage de la date
                            flight_date = row['Slot.Date'].strftime('%d/%m/%Y')
                            
                            # Ligne de texte pour l'e-mail
                            line = f"- Vol du {flight_date}, Immat: {row['Immatriculation']}, CallSign: {row.get('Call sign', 'N/A')}, Slots: {row['Slot.Hour']} & {row['Next_Slot.Hour']} -> Motif: {reason}"
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
    return "Summer" if summer_start <= datetime(year, dt.month, dt.day) < winter_start else "Winter"

def page_saturation_piste(combined_df):
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

    # --- Aggregate data from combined file ---
    df_jour = combined_df[combined_df['Slot.Date'] == jour_choisi].copy()
    
    # Total counts
    total_counts = df_jour.groupby('Heure')[['Vols PPR', 'Vols SCR']].sum()
    
    # Arrival counts
    df_jour_arrivals = df_jour[df_jour['Arrival - Departure'] == 'Arrival']
    arrival_counts = df_jour_arrivals.groupby('Heure')[['Vols PPR', 'Vols SCR']].sum().rename(columns={'Vols PPR': 'Vols PPR Arrivées', 'Vols SCR': 'Vols SCR Arrivées'})

    # --- Combine DataFrames ---
    analysis_df = pd.DataFrame(index=range(24))
    analysis_df = analysis_df.join(capacity_day_df)
    analysis_df = analysis_df.join(total_counts)
    analysis_df = analysis_df.join(arrival_counts)
    analysis_df.fillna(0, inplace=True)

    # --- Calculate Totals and Residuals ---
    analysis_df['Total Vols'] = analysis_df['Vols PPR'] + analysis_df['Vols SCR']
    analysis_df['Total Vols Arrivées'] = analysis_df['Vols PPR Arrivées'] + analysis_df['Vols SCR Arrivées']
    analysis_df['Capacité Résiduelle Totale'] = analysis_df['Capacité Totale'] - analysis_df['Total Vols']
    analysis_df['Capacité Résiduelle Arrivées'] = analysis_df['Capacité Arrivées'] - analysis_df['Total Vols Arrivées']
    analysis_df = analysis_df.astype(int)

    # --- Display UI ---
    analysis_type = st.radio("Choisissez le type d'analyse", ("Totale", "Arrivées"), horizontal=True)
    
    if analysis_type == "Totale":
        value_vars, capacity_col, residual_col = ['Vols PPR', 'Vols SCR'], 'Capacité Totale', 'Capacité Résiduelle Totale'
    else:
        value_vars, capacity_col, residual_col = ['Vols PPR Arrivées', 'Vols SCR Arrivées'], 'Capacité Arrivées', 'Capacité Résiduelle Arrivées'

    source = analysis_df.reset_index().rename(columns={'index': 'Heure'})
    source_melted = source.melt(id_vars=['Heure', capacity_col], value_vars=value_vars, var_name='Type de Vol', value_name='Nombre de Vols')
    
    # Graphique de charge
    bars = alt.Chart(source_melted).mark_bar().encode(x=alt.X('Heure:O', title='Heure'), y=alt.Y('sum(Nombre de Vols):Q', title='Nombre de Vols'), color=alt.Color('Type de Vol:N'), tooltip=['Heure', 'Type de Vol', 'sum(Nombre de Vols)'])
    line = alt.Chart(source).mark_line(color='red', strokeDash=[5,5]).encode(x='Heure:O', y=f'{capacity_col}:Q', tooltip=['Heure', capacity_col])
    charge_chart = (bars + line).properties(title=f"Charge {analysis_type} vs. Capacité").resolve_scale(y='shared')

    # Graphique de capacité résiduelle
    residual_chart = alt.Chart(source).mark_bar().encode(x=alt.X('Heure:O', title='Heure'), y=alt.Y(f'{residual_col}:Q', title='Capacité Résiduelle'), color=alt.condition(alt.datum[residual_col] >= 0, alt.value('green'), alt.value('red')), tooltip=['Heure', residual_col]).properties(title=f"Capacité Résiduelle {analysis_type}")
    
    # Combiner les graphiques verticalement pour aligner les axes
    combined_chart = alt.vconcat(charge_chart, residual_chart)
    
    st.altair_chart(combined_chart, use_container_width=True)

    st.subheader("Détails par heure")
    st.dataframe(analysis_df)

def calculate_hourly_counts(df, analysis_day):
    """Helper function to calculate hourly flight counts from a combined dataframe."""
    df_jour = df[df['Slot.Date'] == analysis_day].copy()
    total_counts = df_jour.groupby('Heure')[['Vols PPR', 'Vols SCR']].sum()
    df_jour_arrivals = df_jour[df_jour['Arrival - Departure'] == 'Arrival']
    arrival_counts = df_jour_arrivals.groupby('Heure')[['Vols PPR', 'Vols SCR']].sum().rename(columns={'Vols PPR': 'Vols PPR Arrivées', 'Vols SCR': 'Vols SCR Arrivées'})
    
    summary_df = pd.DataFrame(index=range(24))
    summary_df = summary_df.join(total_counts)
    summary_df = summary_df.join(arrival_counts)
    summary_df.fillna(0, inplace=True)
    summary_df['Total Vols'] = summary_df['Vols PPR'] + summary_df['Vols SCR']
    summary_df['Total Vols Arrivées'] = summary_df['Vols PPR Arrivées'] + summary_df['Vols SCR Arrivées']
    return summary_df.astype(int)


def page_post_operationnelle():
    """Affiche la page d'analyse comparative Prévu vs. Réel."""
    st.title("🔎 Analyse Post-Opérationnelle")
    st.markdown("Comparez la saturation de piste prévue avec la réalité de la journée passée.")

    jour_analyse = st.date_input("Choisissez la journée à analyser", date.today() - timedelta(days=1))
    
    st.subheader(f"Fichiers pour l'analyse du {jour_analyse.strftime('%d/%m/%Y')}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Fichier de Prévision")
        predicted_file = st.file_uploader("1. Charger le fichier Prévu (PPR+SCR)", type=['xlsx', 'xls'], key="predicted")
    with col2:
        st.markdown("#### Fichier Réalisé")
        actual_file = st.file_uploader("2. Charger le fichier Réalisé (PPR+SCR)", type=['xlsx', 'xls'], key="actual")

    if predicted_file:
        st.session_state.predicted_data = load_and_prepare_data(predicted_file, 'COMBINED')
    if actual_file:
        st.session_state.actual_data = load_and_prepare_data(actual_file, 'COMBINED')

    if st.session_state.predicted_data is not None and st.session_state.actual_data is not None:
        predicted_counts = calculate_hourly_counts(st.session_state.predicted_data, jour_analyse)
        actual_counts = calculate_hourly_counts(st.session_state.actual_data, jour_analyse)

        # Merge and calculate deltas
        comparison_df = predicted_counts.join(actual_counts, lsuffix='_Prévu', rsuffix='_Réalisé')
        comparison_df['Delta PPR'] = comparison_df['Vols PPR_Réalisé'] - comparison_df['Vols PPR_Prévu']
        comparison_df['Delta SCR'] = comparison_df['Vols SCR_Réalisé'] - comparison_df['Vols SCR_Prévu']
        comparison_df['Delta Total'] = comparison_df['Total Vols_Réalisé'] - comparison_df['Total Vols_Prévu']
        
        st.subheader("Tableau Comparatif Prévu vs. Réalisé")
        st.dataframe(comparison_df)

        # --- Visualisation ---
        analysis_type = st.radio("Choisissez le type d'analyse à visualiser", ("Totale", "Arrivées"), horizontal=True, key="post_op_radio")

        if analysis_type == 'Totale':
            melt_vars = ['Total Vols_Prévu', 'Total Vols_Réalisé']
            title = "Comparaison des Vols Totaux (Prévu vs. Réalisé)"
        else:
            melt_vars = ['Total Vols Arrivées_Prévu', 'Total Vols Arrivées_Réalisé']
            title = "Comparaison des Arrivées (Prévu vs. Réalisé)"

        plot_df = comparison_df[melt_vars].reset_index().rename(columns={'index': 'Heure'})
        plot_df = plot_df.melt(id_vars='Heure', var_name='Catégorie', value_name='Nombre de Vols')
        plot_df['Catégorie'] = plot_df['Catégorie'].str.replace('Total Vols_', '').str.replace('Arrivées_', '')

        chart = alt.Chart(plot_df).mark_bar(opacity=0.8).encode(
            x=alt.X('Heure:O', title='Heure'),
            y=alt.Y('Nombre de Vols:Q', title='Nombre de Vols'),
            xOffset='Catégorie:N',
            color='Catégorie:N',
            tooltip=['Heure', 'Catégorie', 'Nombre de Vols']
        ).properties(
            title=title
        ).configure_axis(
            labelAngle=0
        )
        st.altair_chart(chart, use_container_width=True)


# --- Interface principale de l'application ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choisissez une page", ["Détection Doublons", "Analyse & Visualisations", "Analyse de Saturation Piste", "Analyse Post-Opérationnelle"])

st.sidebar.title("Fichiers de données")

# --- Logique de chargement et d'affichage des pages ---
if page in ["Détection Doublons", "Analyse & Visualisations"]:
    ppr_uploaded_file = st.sidebar.file_uploader("Fichier PPR Détaillé (`Reservations.csv`)", type=['csv'])
    if ppr_uploaded_file is not None:
        st.session_state.ppr_data = load_and_prepare_data(ppr_uploaded_file, 'PPR_DETAIL')
    
    if st.session_state.ppr_data is not None:
        if page == "Détection Doublons": page_detection_doublons(st.session_state.ppr_data)
        elif page == "Analyse & Visualisations": page_analyse_visuelle(st.session_state.ppr_data)
    else:
        st.info("Veuillez charger un fichier PPR détaillé. [Lien pour récupérer le fichier](https://ppr.gva.ch/Reservations/Index).")

elif page == "Analyse de Saturation Piste":
    saturation_file = st.sidebar.file_uploader("Fichier Prévisions (PPR+SCR)", type=['xlsx', 'xls'])
    if saturation_file is not None:
        st.session_state.saturation_data = load_and_prepare_data(saturation_file, 'COMBINED')
    
    if st.session_state.saturation_data is not None:
        page_saturation_piste(st.session_state.saturation_data)
    else:
        st.info("Veuillez charger le fichier de prévisions combiné (PPR+SCR).")

elif page == "Analyse Post-Opérationnelle":
    page_post_operationnelle()

