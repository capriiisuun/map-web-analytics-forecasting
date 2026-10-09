import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ==========================================
# 1. Charger les données
# ==========================================

df = pd.read_excel(
    "data/jeu_donnees_analyse_audience_stage.xlsx",
    sheet_name="Sessions"
)


# ==========================================
# 2. Vérification des données
# ==========================================

print("===== APERÇU DES DONNÉES =====")
print(df.head())

print("\n===== COLONNES =====")
print(df.columns.tolist())

print("\nNombre de sessions :", len(df))


# Conversion de la colonne date
df["DebutSession"] = pd.to_datetime(df["DebutSession"])

print("Première date :", df["DebutSession"].min())
print("Dernière date :", df["DebutSession"].max())


# ==========================================
# 3. Nombre de sessions par jour
# ==========================================

sessions_journalieres = (
    df.groupby(df["DebutSession"].dt.date)
    .size()
)

print("\n===== SESSIONS PAR JOUR =====")
print(sessions_journalieres)


# ==========================================
# 4. Visualisation de l'évolution
# ==========================================

plt.figure(figsize=(12, 5))

sessions_journalieres.plot()

plt.title("Évolution quotidienne des sessions")
plt.xlabel("Date")
plt.ylabel("Nombre de sessions")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# ==========================================
# 5. Création du dataset Machine Learning
# ==========================================

data_ml = sessions_journalieres.reset_index()

data_ml.columns = ["Date", "Sessions"]

data_ml["Date"] = pd.to_datetime(data_ml["Date"])


# Jour de la semaine
# 0 = lundi
# 1 = mardi
# ...
# 6 = dimanche

data_ml["jour_semaine"] = (
    data_ml["Date"].dt.dayofweek
)


# Nombre de sessions du jour précédent

data_ml["sessions_j_1"] = (
    data_ml["Sessions"].shift(1)
)


print("\n===== DATASET PRÉPARÉ POUR LE MACHINE LEARNING =====")
print(data_ml.head(10))


# ==========================================
# 6. Préparation du Machine Learning
# ==========================================

# Supprimer la première ligne
# car sessions_j_1 contient NaN

data_ml = data_ml.dropna().copy()


# Variables utilisées pour la prédiction

X = data_ml[
    [
        "jour_semaine",
        "sessions_j_1"
    ]
]


# Variable cible

y = data_ml["Sessions"]


# ==========================================
# 7. Séparation entraînement / test
# ==========================================

# On conserve l'ordre chronologique

taille_train = int(len(data_ml) * 0.8)


X_train = X.iloc[:taille_train]
X_test = X.iloc[taille_train:]

y_train = y.iloc[:taille_train]
y_test = y.iloc[taille_train:]


print("\n===== SÉPARATION DES DONNÉES =====")

print("Taille entraînement :", len(X_train))
print("Taille test :", len(X_test))


# ==========================================
# 8. Création du modèle Random Forest
# ==========================================

modele = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)


# ==========================================
# 9. Entraînement du modèle
# ==========================================

modele.fit(
    X_train,
    y_train
)


print("\n===== MODÈLE ENTRAÎNÉ =====")
print("Random Forest Regressor")


# ==========================================
# 10. Prédictions sur les données de test
# ==========================================

y_pred = modele.predict(X_test)


# ==========================================
# 11. Évaluation du modèle
# ==========================================

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


print("\n===== PERFORMANCE DU MODÈLE =====")

print("MAE :", round(mae, 2))
print("RMSE :", round(rmse, 2))


# ==========================================
# 12. Comparaison réel / prédit
# ==========================================

resultats_test = pd.DataFrame({
    "Date": data_ml.iloc[taille_train:]["Date"].values,
    "Sessions réelles": y_test.values,
    "Sessions prédites": np.round(y_pred, 0)
})


print("\n===== COMPARAISON RÉEL / PRÉDIT =====")
print(resultats_test)


# ==========================================
# 13. Graphique réel / prédit
# ==========================================

plt.figure(figsize=(12, 5))

plt.plot(
    resultats_test["Date"],
    resultats_test["Sessions réelles"],
    marker="o",
    label="Sessions réelles"
)

plt.plot(
    resultats_test["Date"],
    resultats_test["Sessions prédites"],
    marker="o",
    label="Sessions prédites"
)

plt.title(
    "Comparaison des sessions réelles et prédites"
)

plt.xlabel("Date")
plt.ylabel("Nombre de sessions")

plt.xticks(rotation=45)

plt.legend()

plt.tight_layout()

plt.show()


# ==========================================
# 14. Prévision des prochains jours
# ==========================================

print("\n==========================================")
print("PRÉVISION DES PROCHAINS JOURS")
print("==========================================")


# Nombre de jours à prévoir

jours_a_prevoir = 8


# Dernière date disponible

dernier_jour = data_ml["Date"].max()


# Dernier nombre réel de sessions

derniere_valeur = int(
    data_ml.iloc[-1]["Sessions"]
)


previsions = []


# Valeur utilisée pour la prédiction suivante

valeur_precedente = derniere_valeur


# ==========================================
# 15. Génération des prévisions
# ==========================================

for i in range(1, jours_a_prevoir + 1):

    # Date future

    date_future = (
        dernier_jour
        + pd.Timedelta(days=i)
    )


    # Jour de la semaine

    jour_semaine = date_future.dayofweek


    # Création des variables d'entrée

    X_future = pd.DataFrame({
        "jour_semaine": [jour_semaine],
        "sessions_j_1": [valeur_precedente]
    })


    # Prédiction

    prediction = modele.predict(
        X_future
    )[0]


    # Arrondir et éviter les valeurs négatives

    prediction = max(
        0,
        round(prediction)
    )


    # Ajouter la prévision

    previsions.append({
        "Date": date_future,
        "Sessions prévues": prediction
    })


    # La prédiction devient la valeur
    # précédente pour le jour suivant

    valeur_precedente = prediction


# ==========================================
# 16. Tableau des prévisions
# ==========================================

df_previsions = pd.DataFrame(
    previsions
)


print("\n===== PRÉVISIONS =====")

print(df_previsions)


# ==========================================
# 17. Statistiques des prévisions
# ==========================================

moyenne_prevue = (
    df_previsions["Sessions prévues"].mean()
)

jour_max = df_previsions.loc[
    df_previsions["Sessions prévues"].idxmax()
]

jour_min = df_previsions.loc[
    df_previsions["Sessions prévues"].idxmin()
]


print("\n===== STATISTIQUES DES PRÉVISIONS =====")

print(
    "Moyenne prévue :",
    round(moyenne_prevue, 2),
    "sessions"
)

print(
    "Pic prévu :",
    int(jour_max["Sessions prévues"]),
    "sessions le",
    jour_max["Date"].strftime("%d/%m/%Y")
)

print(
    "Minimum prévu :",
    int(jour_min["Sessions prévues"]),
    "sessions le",
    jour_min["Date"].strftime("%d/%m/%Y")
)


# ==========================================
# 18. Graphique historique + prévisions
# ==========================================

plt.figure(figsize=(13, 6))


# Historique

plt.plot(
    data_ml["Date"],
    data_ml["Sessions"],
    marker="o",
    label="Historique"
)


# Prévisions

plt.plot(
    df_previsions["Date"],
    df_previsions["Sessions prévues"],
    marker="o",
    linestyle="--",
    label="Prévisions ML"
)


# Ligne verticale pour séparer
# historique et futur

plt.axvline(
    dernier_jour,
    linestyle=":"
)


plt.title(
    "Historique et prévisions des sessions"
)

plt.xlabel("Date")

plt.ylabel(
    "Nombre de sessions"
)

plt.xticks(
    rotation=45
)

plt.legend()

plt.tight_layout()

plt.show()


# ==========================================
# 19. Sauvegarde des prévisions
# ==========================================

df_previsions.to_csv(
    "previsions_sessions.csv",
    index=False,
    encoding="utf-8-sig"
)


print("\n===== TERMINÉ =====")

print(
    "Les prévisions ont été sauvegardées dans : "
    "previsions_sessions.csv"
)