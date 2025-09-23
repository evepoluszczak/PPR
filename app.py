import streamlit as st
import pandas as pd
from datetime import date, timedelta

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
        # --- 1. Nettoyage et préparation (la normalisation a déjà été faite) ---
        
        # Conversion des types de données
        # Utilisation de errors='coerce' pour transformer les valeurs invalides en NaT (Not a Time) ou NaN
        df['Slot.Date'] = pd.to_datetime(df['Slot.Date'], errors='coerce').dt.date
        df['Date / Heure Creation'] = pd.to_datetime(df['Date / Heure Creation'], errors='coerce')
        # On garde Slot.Hour en tant qu'objet pour l'instant pour éviter les erreurs de conversion de type time
        
        # Suppression des lignes où la date, call sign ou immatriculation sont manquants
        df.dropna(subset=['Slot.Date', 'Call sign', 'Immatriculation'], inplace=True)

        # Filtrer pour ne garder que les PPR actifs (non supprimés)
        active_ppr = df[df['Login (Suppression)'].isnull()].copy()

        # --- 2. Détection des doublons ---

        # Identifier les groupes de doublons potentiels (même date, call sign, immatriculation)
        group_cols = ['Slot.Date', 'Call sign', 'Immatriculation']
        
        # Crée une colonne avec le nombre d'occurrences pour chaque groupe
        active_ppr['Nb de lignes'] = active_ppr.groupby(group_cols)['Slot.Date'].transform('count')
        
        # Filtrer pour ne garder que les groupes avec plus d'une ligne (les vrais doublons)
        duplicates = active_ppr[active_ppr['Nb de lignes'] > 1].copy()

        # Exclure le call sign 'RWYCHK'
        duplicates = duplicates[duplicates['Call sign'] != 'RWYCHK']

        # Filtrer pour les dates J0 et J+1
        today = date.today()
        tomorrow = today + timedelta(days=1)
        duplicates = duplicates[duplicates['Slot.Date'].isin([today, tomorrow])]

        # Appliquer le filtre final de la requête Power Query si la colonne existe
        if 'approach speed Description' in duplicates.columns:
            duplicates = duplicates[duplicates['approach speed Description'] == 'case non cochée']

        # Trier les résultats pour une meilleure lisibilité
        duplicates = duplicates.sort_values(
            by=group_cols + ['Date / Heure Creation'],
            ascending=[True, True, True, True]
        )
        
        return duplicates

    except Exception as e:
        st.error(f"Une erreur est survenue lors du traitement des données : {e}")
        st.warning("Veuillez vérifier que le format du fichier correspond au modèle attendu.")
        return pd.DataFrame()


# --- Interface utilisateur ---

# Composant pour charger le fichier
uploaded_file = st.file_uploader(
    "1. Choisissez le fichier Excel (`PBI_PPR_EPL.xlsx`) ou CSV (`Reservations.csv`)",
    type=['xlsx', 'xls', 'csv']
)

if uploaded_file is not None:
    st.info(f"Fichier chargé : `{uploaded_file.name}`. Traitement en cours...")
    
    raw_df = None
    try:
        # --- Normalisation des données selon le type de fichier ---
        if uploaded_file.name.lower().endswith('.csv'):
            raw_df = pd.read_csv(uploaded_file, sep=';')
            # Renommer les colonnes pour correspondre au format standard
            raw_df = raw_df.rename(columns={
                'CallSign': 'Call sign',
                'Registration': 'Immatriculation'
            })
            # Gérer les dates et heures
            datetime_col = pd.to_datetime(raw_df['Date'], errors='coerce')
            raw_df['Slot.Date'] = datetime_col.dt.date
            raw_df['Slot.Hour'] = datetime_col.dt.time
            # Utiliser la colonne date comme substitut pour la date de création pour le tri
            raw_df['Date / Heure Creation'] = datetime_col
            # Simuler la colonne de suppression pour la logique de filtrage
            raw_df['Login (Suppression)'] = None
            raw_df.loc[raw_df['Deleted'] == True, 'Login (Suppression)'] = 'Deleted'

        elif uploaded_file.name.lower().endswith(('.xls', '.xlsx')):
            excel_df = pd.read_excel(uploaded_file, sheet_name="PBI_PPR_J0_J5", header=None)
            # Promotion des en-têtes
            excel_df.columns = excel_df.iloc[0]
            raw_df = excel_df.drop(excel_df.index[0]).reset_index(drop=True)
            # Renommage des colonnes
            raw_df = raw_df.rename(columns={
                "Slot Réservation.Date": "Slot.Date",
                "Heure": "Slot.Hour"
            })

    except Exception as e:
        if "Worksheet named 'PBI_PPR_J0_J5' not found" in str(e):
              st.error("Erreur : La feuille nommée `PBI_PPR_J0_J5` n'a pas été trouvée. Veuillez vérifier le fichier Excel.")
        else:
            st.error(f"Erreur lors de la lecture du fichier : {e}")
        raw_df = None # Assure que le traitement ne continue pas en cas d'erreur de lecture
    
    if raw_df is not None:
        # Traitement des données
        result_df = process_ppr_data(raw_df)

        st.header("2. Résultats de l'analyse")

        if not result_df.empty:
            # Calculer le nombre de groupes uniques de doublons
            group_cols = ['Slot.Date', 'Call sign', 'Immatriculation']
            num_groups = len(result_df.groupby(group_cols))

            st.success(f"**{num_groups}** groupe(s) de doublons détecté(s) !")
            
            # Affichage des résultats
            st.dataframe(result_df)
            
            st.markdown("---")
            st.write("### Actions recommandées :")
            st.write("- **Vérifiez** chaque groupe de lignes ci-dessus.")
            st.write("- **Contactez** les agents handling ou les utilisateurs (`Login`) concernés pour clarifier la situation.")
            st.write("- **Demandez** la suppression de la réservation incorrecte afin de régulariser la planification.")

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
            st.success("🎉 Aucune PPR en doublon détectée pour aujourd'hui et demain selon les critères définis.")
else:
    st.info("En attente du chargement d'un fichier.")

