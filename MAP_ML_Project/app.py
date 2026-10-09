import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MAP - Audience Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. STYLE GÉNÉRAL
# ============================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

section[data-testid="stSidebar"] {
    border-right: 1px solid #e5e7eb;
}

h1 {
    font-weight: 700;
}

h2 {
    font-weight: 650;
}

h3 {
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. CHARGEMENT DES DONNÉES
# ============================================================

@st.cache_data
def charger_sessions():

    df = pd.read_excel(
        "data/jeu_donnees_analyse_audience_stage.xlsx",
        sheet_name="Sessions"
    )

    df["DebutSession"] = pd.to_datetime(
        df["DebutSession"],
        errors="coerce"
    )

    return df


@st.cache_data
def charger_evenements():

    return pd.read_excel(
        "data/jeu_donnees_analyse_audience_stage.xlsx",
        sheet_name="Evenements"
    )


@st.cache_data
def charger_contenus():

    return pd.read_excel(
        "data/jeu_donnees_analyse_audience_stage.xlsx",
        sheet_name="Contenus"
    )


@st.cache_data
def charger_visiteurs():

    return pd.read_excel(
        "data/jeu_donnees_analyse_stage.xlsx"
        if False
        else "data/jeu_donnees_analyse_audience_stage.xlsx",
        sheet_name="Visiteurs"
    )


df = charger_sessions()
evenements = charger_evenements()
contenus = charger_contenus()
visiteurs = charger_visiteurs()


# ============================================================
# 4. NETTOYAGE
# ============================================================

if "DureeSessionSecondes" in df.columns:

    df["DureeSessionSecondes"] = pd.to_numeric(
        df["DureeSessionSecondes"],
        errors="coerce"
    )


if "PagesParSession" in df.columns:

    df["PagesParSession"] = pd.to_numeric(
        df["PagesParSession"],
        errors="coerce"
    )


# ============================================================
# 5. SESSIONS JOURNALIÈRES
# ============================================================

sessions_journalieres = (
    df.groupby(
        df["DebutSession"].dt.date
    )
    .size()
    .reset_index(name="Sessions")
)

sessions_journalieres.columns = [
    "Date",
    "Sessions"
]

sessions_journalieres["Date"] = pd.to_datetime(
    sessions_journalieres["Date"]
)

sessions_journalieres["jour_semaine"] = (
    sessions_journalieres["Date"].dt.dayofweek
)

sessions_journalieres["sessions_j_1"] = (
    sessions_journalieres["Sessions"].shift(1)
)


# ============================================================
# 6. MODÈLE MACHINE LEARNING
# ============================================================

@st.cache_resource
def entrainer_modele():

    data_ml = sessions_journalieres.dropna().copy()

    X = data_ml[
        [
            "jour_semaine",
            "sessions_j_1"
        ]
    ]

    y = data_ml["Sessions"]

    taille_train = int(
        len(data_ml) * 0.8
    )

    X_train = X.iloc[:taille_train]
    X_test = X.iloc[taille_train:]

    y_train = y.iloc[:taille_train]
    y_test = y.iloc[taille_train:]

    modele = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    modele.fit(
        X_train,
        y_train
    )

    y_pred = modele.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    return modele, mae, rmse


modele, mae, rmse = entrainer_modele()


# ============================================================
# 7. FONCTION DE PRÉVISION
# ============================================================

def generer_previsions(
    modele,
    sessions_journalieres,
    nombre_jours
):

    dernier_jour = (
        sessions_journalieres["Date"].max()
    )

    derniere_valeur = int(
        sessions_journalieres.iloc[-1]["Sessions"]
    )

    previsions = []

    valeur_precedente = derniere_valeur

    for i in range(
        1,
        nombre_jours + 1
    ):

        date_future = (
            dernier_jour
            + pd.Timedelta(days=i)
        )

        jour_semaine = (
            date_future.dayofweek
        )

        X_future = pd.DataFrame({
            "jour_semaine": [
                jour_semaine
            ],
            "sessions_j_1": [
                valeur_precedente
            ]
        })

        prediction = modele.predict(
            X_future
        )[0]

        prediction = max(
            0,
            round(prediction)
        )

        previsions.append({
            "Date": date_future,
            "Sessions prévues": prediction
        })

        valeur_precedente = prediction

    return pd.DataFrame(
        previsions
    )


# ============================================================
# 8. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            padding:10px 0 15px 0;
            font-size:30px;
            font-weight:700;
        ">
            📊 MAP
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="
            font-size:16px;
            font-weight:600;
            line-height:1.5;
        ">
            Audience Analytics &<br>
            Machine Learning
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Vue générale",
            "📈 Analyse audience",
            "🎯 Comportement utilisateurs",
            "📰 Performance des contenus",
            "🔮 Prévisions ML",
            "📁 Données"
        ]
    )

    st.divider()

    st.caption(
        "Projet d'analyse d'audience web"
    )

    st.caption(
        "Python • Machine Learning • Streamlit • Power BI"
    )


# ============================================================
# PAGE 1 — VUE GÉNÉRALE
# ============================================================

if page == "🏠 Vue générale":

    st.title(
        "📊 Dashboard Audience MAP"
    )

    st.markdown(
        "### Vue générale de l'activité web"
    )

    st.info(
        "Cette application permet d'analyser l'audience web, "
        "d'étudier le comportement des utilisateurs, "
        "d'évaluer la performance des contenus et de prévoir "
        "l'évolution du trafic grâce au Machine Learning."
    )

    st.write("")

    # ========================================================
    # CALCUL DES KPI
    # ========================================================

    total_sessions = len(df)

    visiteurs_uniques = (
        df["VisiteurId"]
        .nunique()
    )

    total_conversions = (
        df["Conversion"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("oui")
        .sum()
    )

    duree_moyenne = (
        df["DureeSessionSecondes"]
        .mean()
    )

    # ========================================================
    # CARTES KPI
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.html(
            f"""
            <div style="
                background:white;
                padding:22px;
                border-radius:16px;
                border:1px solid #e5e7eb;
                min-height:135px;
                box-shadow:0 2px 8px rgba(0,0,0,0.05);
            ">

                <div style="
                    font-size:15px;
                    color:#6b7280;
                ">
                    👥 Visiteurs uniques
                </div>

                <div style="
                    font-size:32px;
                    font-weight:700;
                    color:#5b35a8;
                    margin-top:10px;
                ">
                    {visiteurs_uniques:,}
                </div>

                <div style="
                    font-size:13px;
                    color:#8b8f99;
                    margin-top:5px;
                ">
                    Audience observée
                </div>

            </div>
            """
        )

    with col2:

        st.html(
            f"""
            <div style="
                background:white;
                padding:22px;
                border-radius:16px;
                border:1px solid #e5e7eb;
                min-height:135px;
                box-shadow:0 2px 8px rgba(0,0,0,0.05);
            ">

                <div style="
                    font-size:15px;
                    color:#6b7280;
                ">
                    🌐 Sessions
                </div>

                <div style="
                    font-size:32px;
                    font-weight:700;
                    color:#3987e8;
                    margin-top:10px;
                ">
                    {total_sessions:,}
                </div>

                <div style="
                    font-size:13px;
                    color:#8b8f99;
                    margin-top:5px;
                ">
                    Sessions enregistrées
                </div>

            </div>
            """
        )

    with col3:

        st.html(
            f"""
            <div style="
                background:white;
                padding:22px;
                border-radius:16px;
                border:1px solid #e5e7eb;
                min-height:135px;
                box-shadow:0 2px 8px rgba(0,0,0,0.05);
            ">

                <div style="
                    font-size:15px;
                    color:#6b7280;
                ">
                    🎯 Conversions
                </div>

                <div style="
                    font-size:32px;
                    font-weight:700;
                    color:#e13d68;
                    margin-top:10px;
                ">
                    {total_conversions:,}
                </div>

                <div style="
                    font-size:13px;
                    color:#8b8f99;
                    margin-top:5px;
                ">
                    Conversions enregistrées
                </div>

            </div>
            """
        )

    with col4:

        st.html(
            f"""
            <div style="
                background:white;
                padding:22px;
                border-radius:16px;
                border:1px solid #e5e7eb;
                min-height:135px;
                box-shadow:0 2px 8px rgba(0,0,0,0.05);
            ">

                <div style="
                    font-size:15px;
                    color:#6b7280;
                ">
                    ⏱️ Durée moyenne
                </div>

                <div style="
                    font-size:32px;
                    font-weight:700;
                    color:#7653bd;
                    margin-top:10px;
                ">
                    {duree_moyenne:.0f}s
                </div>

                <div style="
                    font-size:13px;
                    color:#8b8f99;
                    margin-top:5px;
                ">
                    Par session
                </div>

            </div>
            """
        )

    st.write("")

    # ========================================================
    # GRAPHIQUE PRINCIPAL
    # ========================================================

    st.subheader(
        "📈 Évolution quotidienne des sessions"
    )

    fig = px.line(
        sessions_journalieres,
        x="Date",
        y="Sessions",
        markers=True
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Nombre de sessions",
        hovermode="x unified",
        height=430
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ========================================================
    # INDICATEURS COMPLÉMENTAIRES
    # ========================================================

    st.subheader(
        "📊 Indicateurs complémentaires"
    )

    pages_moyennes = (
        df["PagesParSession"]
        .mean()
    )

    taux_rebond = (
        df["Rebond"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("oui")
        .mean()
        * 100
    )

    taux_conversion = (
        df["Conversion"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("oui")
        .mean()
        * 100
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "📄 Pages / session",
            f"{pages_moyennes:.2f}"
        )

    with col2:
        st.metric(
            "↩️ Taux de rebond",
            f"{taux_rebond:.1f}%"
        )

    with col3:
        st.metric(
            "🎯 Taux de conversion",
            f"{taux_conversion:.1f}%"
        )


# ============================================================
# PAGE 2 — ANALYSE AUDIENCE
# ============================================================

elif page == "📈 Analyse audience":

    st.title(
        "📈 Analyse de l'audience"
    )

    st.markdown(
        "### Exploration des principales dimensions de l'audience"
    )

    col1, col2 = st.columns(2)

    with col1:

        source_counts = (
            df["SourceTrafic"]
            .value_counts()
            .reset_index()
        )

        source_counts.columns = [
            "Source",
            "Sessions"
        ]

        fig_source = px.bar(
            source_counts,
            x="Source",
            y="Sessions",
            text_auto=True,
            title="Sessions par source de trafic"
        )

        fig_source.update_layout(
            xaxis_title="Source de trafic",
            yaxis_title="Sessions"
        )

        st.plotly_chart(
            fig_source,
            use_container_width=True
        )

    with col2:

        support_counts = (
            df["Support"]
            .value_counts()
            .reset_index()
        )

        support_counts.columns = [
            "Support",
            "Sessions"
        ]

        fig_support = px.pie(
            support_counts,
            names="Support",
            values="Sessions",
            hole=0.45,
            title="Répartition des sessions par support"
        )

        st.plotly_chart(
            fig_support,
            use_container_width=True
        )

    st.subheader(
        "🌍 Sessions par région"
    )

    region_counts = (
        df["Region"]
        .value_counts()
        .reset_index()
    )

    region_counts.columns = [
        "Région",
        "Sessions"
    ]

    fig_region = px.bar(
        region_counts,
        x="Région",
        y="Sessions",
        text_auto=True,
        title="Audience par région"
    )

    st.plotly_chart(
        fig_region,
        use_container_width=True
    )

    st.subheader(
        "🎯 Analyse des conversions"
    )

    conversion_counts = (
        df["Conversion"]
        .astype(str)
        .str.strip()
        .value_counts()
        .reset_index()
    )

    conversion_counts.columns = [
        "Conversion",
        "Nombre"
    ]

    fig_conversion = px.pie(
        conversion_counts,
        names="Conversion",
        values="Nombre",
        hole=0.45,
        title="Sessions avec / sans conversion"
    )

    st.plotly_chart(
        fig_conversion,
        use_container_width=True
    )


# ============================================================
# PAGE 3 — COMPORTEMENT UTILISATEURS
# ============================================================

elif page == "🎯 Comportement utilisateurs":

    st.title(
        "🎯 Comportement des utilisateurs"
    )

    st.markdown(
        "### Comprendre comment les visiteurs utilisent le site"
    )

    # ========================================================
    # KPI
    # ========================================================

    duree = pd.to_numeric(
        df["DureeSessionSecondes"],
        errors="coerce"
    )

    pages = pd.to_numeric(
        df["PagesParSession"],
        errors="coerce"
    )

    rebond = (
        df["Rebond"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    conversion = (
        df["Conversion"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    taux_rebond = (
        rebond.eq("oui").mean()
        * 100
    )

    taux_conversion = (
        conversion.eq("oui").mean()
        * 100
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "⏱️ Durée moyenne",
            f"{duree.mean():.0f} sec"
        )

    with col2:
        st.metric(
            "📄 Pages / session",
            f"{pages.mean():.2f}"
        )

    with col3:
        st.metric(
            "↩️ Taux de rebond",
            f"{taux_rebond:.1f}%"
        )

    with col4:
        st.metric(
            "🎯 Taux de conversion",
            f"{taux_conversion:.1f}%"
        )

    st.write("")

    # ========================================================
    # TYPE DE VISITEUR
    # ========================================================

    st.subheader(
        "👤 Type de visiteur"
    )

    if "TypeVisiteur" in visiteurs.columns:

        visiteurs_type = (
            visiteurs["TypeVisiteur"]
            .fillna("Non renseigné")
            .value_counts()
            .reset_index()
        )

        visiteurs_type.columns = [
            "Type",
            "Visiteurs"
        ]

        fig_type = px.pie(
            visiteurs_type,
            names="Type",
            values="Visiteurs",
            hole=0.45,
            title="Répartition des visiteurs"
        )

        st.plotly_chart(
            fig_type,
            use_container_width=True
        )

    else:

        st.info(
            "La colonne TypeVisiteur n'est pas disponible."
        )

    # ========================================================
    # SUPPORT
    # ========================================================

    st.subheader(
        "📱 Comportement selon le support"
    )

    support_behavior = (
        df.groupby("Support")
        .agg(
            Sessions=("SessionId", "count"),
            DureeMoyenne=(
                "DureeSessionSecondes",
                "mean"
            ),
            PagesMoyennes=(
                "PagesParSession",
                "mean"
            )
        )
        .reset_index()
    )

    col1, col2 = st.columns(2)

    with col1:

        fig_duree = px.bar(
            support_behavior,
            x="Support",
            y="DureeMoyenne",
            text_auto=".1f",
            title="Durée moyenne par support"
        )

        fig_duree.update_layout(
            yaxis_title="Durée moyenne (secondes)"
        )

        st.plotly_chart(
            fig_duree,
            use_container_width=True
        )

    with col2:

        fig_pages = px.bar(
            support_behavior,
            x="Support",
            y="PagesMoyennes",
            text_auto=".2f",
            title="Pages moyennes par support"
        )

        fig_pages.update_layout(
            yaxis_title="Pages par session"
        )

        st.plotly_chart(
            fig_pages,
            use_container_width=True
        )

    # ========================================================
    # SOURCE DE TRAFIC
    # ========================================================

    st.subheader(
        "🌐 Qualité des sources de trafic"
    )

    source_behavior = (
        df.groupby("SourceTrafic")
        .agg(
            Sessions=("SessionId", "count"),
            DureeMoyenne=(
                "DureeSessionSecondes",
                "mean"
            ),
            PagesMoyennes=(
                "PagesParSession",
                "mean"
            )
        )
        .reset_index()
    )

    fig_source = px.scatter(
        source_behavior,
        x="DureeMoyenne",
        y="PagesMoyennes",
        size="Sessions",
        text="SourceTrafic",
        title="Qualité des sources de trafic"
    )

    fig_source.update_traces(
        textposition="top center"
    )

    st.plotly_chart(
        fig_source,
        use_container_width=True
    )

    # ========================================================
    # REGION
    # ========================================================

    st.subheader(
        "🌍 Comportement par région"
    )

    region_behavior = (
        df.groupby("Region")
        .agg(
            Sessions=("SessionId", "count"),
            DureeMoyenne=(
                "DureeSessionSecondes",
                "mean"
            ),
            PagesMoyennes=(
                "PagesParSession",
                "mean"
            )
        )
        .reset_index()
    )

    st.dataframe(
        region_behavior.round(2),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PAGE 4 — PERFORMANCE DES CONTENUS
# ============================================================

elif page == "📰 Performance des contenus":

    st.title(
        "📰 Performance des contenus"
    )

    st.markdown(
        "### Identifier les contenus qui attirent le plus l'audience"
    )

    st.info(
        "Cette section permet d'étudier les interactions avec les contenus "
        "afin d'identifier les contenus les plus consultés et les plus performants."
    )

    st.write("")

    # ========================================================
    # PRÉPARATION
    # ========================================================

    contenu_stats = (
        evenements
        .groupby(
            ["PageId", "Titre"],
            dropna=False
        )
        .size()
        .reset_index(
            name="Interactions"
        )
    )

    contenu_stats["Titre"] = (
        contenu_stats["Titre"]
        .fillna("Contenu sans titre")
        .astype(str)
    )

    contenu_stats = (
        contenu_stats
        .sort_values(
            "Interactions",
            ascending=False
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # TOP 10
    # ========================================================

    st.subheader(
        "🏆 Top des contenus"
    )

    col1, col2 = st.columns(
        [1.3, 1]
    )

    with col1:

        top10 = contenu_stats.head(10).copy()

        top10 = top10.sort_values(
            "Interactions",
            ascending=True
        )

        fig_top = px.bar(
            top10,
            x="Interactions",
            y="Titre",
            orientation="h",
            text="Interactions",
            title="Top 10 des contenus les plus consultés"
        )

        fig_top.update_traces(
            textposition="outside"
        )

        fig_top.update_layout(
            xaxis_title="Interactions",
            yaxis_title="Contenu",
            showlegend=False
        )

        st.plotly_chart(
            fig_top,
            use_container_width=True
        )

    with col2:

        st.markdown(
            "#### 📊 Classement des contenus"
        )

        classement = contenu_stats.copy()

        classement.insert(
            0,
            "Rang",
            range(
                1,
                len(classement) + 1
            )
        )

        classement = classement[
            [
                "Rang",
                "Titre",
                "Interactions"
            ]
        ]

        classement.columns = [
            "Rang",
            "Contenu",
            "Interactions"
        ]

        st.dataframe(
            classement.head(10),
            use_container_width=True,
            hide_index=True,
            height=430
        )

    # ========================================================
    # TYPES D'INTERACTIONS
    # ========================================================

    st.subheader(
        "👀 Types d'interactions"
    )

    colonne_type = None

    for colonne in evenements.columns:

        nom = str(colonne).lower()

        if (
            "typeevenement" in nom
            or "type_evenement" in nom
            or "type evenement" in nom
            or nom == "type"
            or nom == "eventtype"
            or nom == "event_type"
        ):

            colonne_type = colonne
            break

    if colonne_type is not None:

        interactions = (
            evenements[
                colonne_type
            ]
            .fillna("Autre")
            .astype(str)
            .value_counts()
            .reset_index()
        )

        interactions.columns = [
            "Type",
            "Nombre"
        ]

        fig_interactions = px.pie(
            interactions,
            names="Type",
            values="Nombre",
            hole=0.45,
            title="Répartition des interactions"
        )

        st.plotly_chart(
            fig_interactions,
            use_container_width=True
        )

    else:

        st.info(
            "La colonne de type d'interaction n'a pas été détectée."
        )

    # ========================================================
    # TYPE DE CONTENU
    # ========================================================

    st.subheader(
        "📰 Performance par type de contenu"
    )

    if "TypeContenu" in evenements.columns:

        type_contenu = (
            evenements
            .groupby("TypeContenu")
            .size()
            .reset_index(
                name="Interactions"
            )
            .sort_values(
                "Interactions",
                ascending=False
            )
        )

        fig_type = px.bar(
            type_contenu,
            x="TypeContenu",
            y="Interactions",
            text="Interactions",
            title="Interactions par type de contenu"
        )

        fig_type.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            fig_type,
            use_container_width=True
        )

    # ========================================================
    # RUBRIQUE
    # ========================================================

    st.subheader(
        "📂 Performance par rubrique"
    )

    if "Rubrique" in evenements.columns:

        rubrique_stats = (
            evenements
            .groupby("Rubrique")
            .size()
            .reset_index(
                name="Interactions"
            )
            .sort_values(
                "Interactions",
                ascending=False
            )
        )

        fig_rubrique = px.bar(
            rubrique_stats,
            x="Rubrique",
            y="Interactions",
            text="Interactions",
            title="Interactions par rubrique"
        )

        fig_rubrique.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            fig_rubrique,
            use_container_width=True
        )

    # ========================================================
    # THEME
    # ========================================================

    st.subheader(
        "🏷️ Performance par thème"
    )

    if "Theme" in evenements.columns:

        theme_stats = (
            evenements
            .groupby("Theme")
            .size()
            .reset_index(
                name="Interactions"
            )
            .sort_values(
                "Interactions",
                ascending=False
            )
        )

        fig_theme = px.bar(
            theme_stats,
            x="Theme",
            y="Interactions",
            text="Interactions",
            title="Interactions par thème"
        )

        fig_theme.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            fig_theme,
            use_container_width=True
        )

    # ========================================================
    # OBSERVATIONS
    # ========================================================

    st.subheader(
        "💡 Principales observations"
    )

    if len(contenu_stats) > 0:

        meilleur = contenu_stats.iloc[0]

        st.success(
            f"🏆 Contenu le plus performant : "
            f"« {meilleur['Titre']} » avec "
            f"{int(meilleur['Interactions'])} interactions."
        )

        if len(contenu_stats) >= 2:

            deuxieme = contenu_stats.iloc[1]

            st.info(
                f"🥈 Deuxième contenu le plus consulté : "
                f"« {deuxieme['Titre']} » avec "
                f"{int(deuxieme['Interactions'])} interactions."
            )

    # ========================================================
    # TELECHARGEMENT
    # ========================================================

    csv_contenus = contenu_stats.to_csv(
        index=False,
        encoding="utf-8-sig"
    )

    st.download_button(
        "⬇️ Télécharger le classement des contenus",
        data=csv_contenus,
        file_name="performance_contenus.csv",
        mime="text/csv"
    )


# ============================================================
# PAGE 5 — PRÉVISIONS ML
# ============================================================

elif page == "🔮 Prévisions ML":

    st.title(
        "🔮 Prévisions Machine Learning"
    )

    st.markdown(
        "### Anticipation du trafic web"
    )

    st.info(
        "Le modèle Random Forest utilise le jour de la semaine "
        "et le nombre de sessions du jour précédent afin d'estimer "
        "le trafic des prochains jours."
    )

    st.write("")

    # ========================================================
    # HORIZON
    # ========================================================

    jours_prevision = st.slider(
        "📅 Horizon de prévision",
        min_value=1,
        max_value=30,
        value=8,
        step=1
    )

    df_previsions = generer_previsions(
        modele,
        sessions_journalieres,
        jours_prevision
    )

    moyenne_prevue = (
        df_previsions[
            "Sessions prévues"
        ].mean()
    )

    total_prevu = (
        df_previsions[
            "Sessions prévues"
        ].sum()
    )

    jour_max = df_previsions.loc[
        df_previsions[
            "Sessions prévues"
        ].idxmax()
    ]

    jour_min = df_previsions.loc[
        df_previsions[
            "Sessions prévues"
        ].idxmin()
    ]

    # ========================================================
    # KPI
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "📊 Moyenne prévue",
            f"{moyenne_prevue:.1f}"
        )

    with col2:

        st.metric(
            "🌐 Total prévu",
            f"{int(total_prevu)}"
        )

    with col3:

        st.metric(
            "⬆️ Pic prévu",
            f"{int(jour_max['Sessions prévues'])}",
            jour_max["Date"].strftime("%d/%m/%Y")
        )

    with col4:

        st.metric(
            "⬇️ Minimum prévu",
            f"{int(jour_min['Sessions prévues'])}",
            jour_min["Date"].strftime("%d/%m/%Y")
        )

    st.write("")

    # ========================================================
    # GRAPHIQUE
    # ========================================================

    st.subheader(
        "📈 Historique et prévisions"
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=sessions_journalieres["Date"],
            y=sessions_journalieres["Sessions"],
            mode="lines+markers",
            name="Historique"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_previsions["Date"],
            y=df_previsions["Sessions prévues"],
            mode="lines+markers",
            name="Prévisions ML",
            line=dict(
                dash="dash"
            )
        )
    )

    fig.update_layout(
        title="Sessions historiques et sessions prévues",
        xaxis_title="Date",
        yaxis_title="Nombre de sessions",
        hovermode="x unified",
        height=450
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ========================================================
    # TABLEAU
    # ========================================================

    st.subheader(
        "📅 Prévisions détaillées"
    )

    tableau = df_previsions.copy()

    tableau["Date"] = (
        tableau["Date"]
        .dt.strftime("%d/%m/%Y")
    )

    st.dataframe(
        tableau,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # INTERPRÉTATION
    # ========================================================

    st.subheader(
        "💡 Interprétation automatique"
    )

    moyenne_historique = (
        sessions_journalieres[
            "Sessions"
        ].mean()
    )

    pourcentage = (
        (
            moyenne_prevue
            - moyenne_historique
        )
        / moyenne_historique
    ) * 100

    if pourcentage > 5:

        st.success(
            f"📈 Le trafic prévu est supérieur à la moyenne "
            f"historique d'environ {pourcentage:.1f}%."
        )

    elif pourcentage < -5:

        st.warning(
            f"📉 Le trafic prévu est inférieur à la moyenne "
            f"historique d'environ {abs(pourcentage):.1f}%."
        )

    else:

        st.info(
            "➡️ Le trafic prévu reste globalement proche "
            "de la moyenne historique."
        )

    # ========================================================
    # TELECHARGEMENT
    # ========================================================

    csv = df_previsions.to_csv(
        index=False,
        encoding="utf-8-sig"
    )

    st.download_button(
        "⬇️ Télécharger les prévisions CSV",
        data=csv,
        file_name="previsions_sessions.csv",
        mime="text/csv"
    )


# ============================================================
# PAGE 6 — DONNÉES
# ============================================================

elif page == "📁 Données":

    st.title(
        "📁 Données"
    )

    st.markdown(
        "### Exploration du jeu de données utilisé"
    )

    # ========================================================
    # KPI
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "🌐 Sessions",
            f"{len(df):,}"
        )

    with col2:

        st.metric(
            "📋 Colonnes",
            len(df.columns)
        )

    with col3:

        st.metric(
            "📅 Date début",
            df["DebutSession"]
            .min()
            .strftime("%d/%m/%Y")
        )

    with col4:

        st.metric(
            "📅 Date fin",
            df["DebutSession"]
            .max()
            .strftime("%d/%m/%Y")
        )

    st.write("")

    # ========================================================
    # FILTRES
    # ========================================================

    st.subheader(
        "🔎 Filtrer les données"
    )

    col1, col2 = st.columns(2)

    with col1:

        sources = st.multiselect(
            "Source de trafic",
            options=sorted(
                df["SourceTrafic"]
                .dropna()
                .unique()
            )
        )

    with col2:

        supports = st.multiselect(
            "Support",
            options=sorted(
                df["Support"]
                .dropna()
                .unique()
            )
        )

    df_filtre = df.copy()

    if sources:

        df_filtre = df_filtre[
            df_filtre[
                "SourceTrafic"
            ].isin(sources)
        ]

    if supports:

        df_filtre = df_filtre[
            df_filtre[
                "Support"
            ].isin(supports)
        ]

    st.write(
        f"**{len(df_filtre):,}** sessions affichées"
    )

    st.dataframe(
        df_filtre,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # TELECHARGEMENT
    # ========================================================

    csv_donnees = df_filtre.to_csv(
        index=False,
        encoding="utf-8-sig"
    )

    st.download_button(
        "⬇️ Télécharger les données filtrées",
        data=csv_donnees,
        file_name="donnees_sessions_filtrees.csv",
        mime="text/csv"
    )