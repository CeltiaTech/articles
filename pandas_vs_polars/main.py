import time
import pandas as pd

# Début du chrono
start = time.perf_counter()

# Lecture du CSV en ne chargeant que les colonnes utiles
df = pd.read_csv(
    "PlayerStatistics.csv",
    usecols=[
        "personId",
        "firstName",
        "lastName",
        "gameType",
        "threePointersMade",
    ],
    dtype={
        "threePointersMade": "float32",
    },
)

# Filtrage des types de matchs
filtered_df = df[
    df["gameType"].isin([
        "Regular Season",
        "NBA Emirates Cup",
    ])
]

# Groupement par joueur
top_shooters = (
    filtered_df
    .groupby("personId", as_index=False)
    .agg({
        "threePointersMade": "sum",
        "firstName": "first",
        "lastName": "first",
    })
)

# Renommage de la colonne agrégée
top_shooters = top_shooters.rename(
    columns={
        "threePointersMade": "career_3pm"
    }
)

# Création du nom complet
top_shooters["fullName"] = (
    top_shooters["firstName"]
    + " "
    + top_shooters["lastName"]
)

# Sélection des colonnes finales
top_shooters = top_shooters[
    ["fullName", "career_3pm"]
]

# Tri décroissant
top_shooters = top_shooters.sort_values(
    by="career_3pm",
    ascending=False,
)

# Top 20
top_shooters = top_shooters.head(20)
# Temps d'exécution
duration = time.perf_counter() - start
print(f"Temps d'exécution: {duration:.6f}s")
# Ajout d'un index commençant à 1
top_shooters.insert(
    0,
    "index",
    range(1, len(top_shooters) + 1)
)

# Affichage
for _, row in top_shooters.iterrows():
    print(
        f"{row['index']}. "
        f"{row['fullName']}: "
        f"{row['career_3pm']}"
    )

