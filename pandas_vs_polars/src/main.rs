// Importation du module prelude de Polars contenant les structures et fonctions principales.
use polars::prelude::*;

// Importation d'un utilitaire permettant d'itérer facilement sur les lignes d'un DataFrame Polars.
use polars_rows_iter::*;

// Importation de Instant pour mesurer le temps d'exécution du programme.
use std::time::Instant;

// Fonction principale du programme.
// Retourne un Result afin de propager les erreurs automatiquement avec ?.
fn main() -> Result<(), Box<dyn std::error::Error>> {

    // Enregistre le moment exact du démarrage du programme.
    // Servira à calculer le temps total d'exécution.
    let start = Instant::now();

    // Définition manuelle du type d'une colonne du CSV.
    // Cela évite les erreurs d'auto-détection de Polars.
    let schema_overrides = Schema::from_iter([
        Field::new("personId".into(), DataType::Int32),
        Field::new("firstName".into(), DataType::String),
        Field::new("lastName".into(), DataType::String),
        Field::new("gameType".into(), DataType::String),
        // Déclare que la colonne "threePointersMade"
        // doit être interprétée comme un Float32.
        Field::new("threePointersMade".into(), DataType::Float32),
    ]);

    // Lecture du fichier CSV et création d'un DataFrame.
    let df = CsvReadOptions::default()

        // Remplace les types détectés automatiquement
        // par le schéma défini précédemment.
        .with_schema_overwrite(Some(Arc::new(schema_overrides)))

        // Sélectionne uniquement les colonnes utiles
        // afin d'éviter de charger tout le CSV en mémoire.
        .with_columns(Some(Arc::new([

            // Identifiant unique du joueur.
            "personId".into(),

            // Prénom du joueur.
            "firstName".into(),

            // Nom du joueur.
            "lastName".into(),

            // Type de match (Regular Season, Playoffs, etc.).
            "gameType".into(),

            // Nombre de tirs à 3 points marqués.
            "threePointersMade".into(),
        ])))

        // Spécifie le chemin du fichier CSV à lire.
        .try_into_reader_with_file_path(Some("PlayerStatistics.csv".into()))?

        // Exécute la lecture du CSV et retourne le DataFrame final.
        .finish()?; 
  
    // Création d'un DataFrame contenant les meilleurs shooteurs à 3 points.
    let top_shooters = df

        // Passe en mode LazyFrame pour optimiser les calculs.
        .lazy()

        // Filtre les lignes selon le type de match.
        .filter(

            // Sélectionne la colonne "gameType".
            col("gameType")

                // Vérifie si gameType appartient à une liste de valeurs autorisées.
                .is_in(

                    // Création d'une Series contenant les types de matchs à conserver.
                    lit(Series::new(

                        // Nom de la série.
                        "types".into(),

                        // Types de matchs conservés.
                        [
                            "Regular Season",
                            "NBA Emirates Cup",
                        ],
                    )),

                    // false = comportement par défaut concernant les valeurs nulles.
                    false,
                )
        )

        // Groupe les lignes par joueur grâce à personId.
        .group_by([
            col("personId")
        ])

        // Applique des agrégations sur chaque groupe.
        .agg([

            // Calcule la somme totale des tirs à 3 points réussis.
            col("threePointersMade")

                // Additionne toutes les valeurs du groupe.
                .sum()

                // Renomme la colonne résultante.
                .alias("career_3pm"),

            // Récupère le premier prénom du groupe.
            // Comme le joueur est identique dans le groupe,
            // cela sert simplement à conserver l'information.
            col("firstName").first(),

            // Même logique pour le nom de famille.
            col("lastName").first()
        ])

        // Ajoute une nouvelle colonne calculée.
        .with_columns([

            // Concatène prénom + espace + nom.
            (
                col("firstName")
                + lit(" ")
                + col("lastName")
            )

            // Renomme cette nouvelle colonne.
            .alias("fullName"),

        ])

        // Sélectionne uniquement les colonnes utiles.
        .select([

            // Nom complet du joueur.
            col("fullName"),

            // Total de tirs à 3 points.
            col("career_3pm"),
        ])

        // Trie les résultats.
        .sort(

            // Tri sur la colonne career_3pm.
            ["career_3pm"],

            // Options de tri.
            SortMultipleOptions::default()

                // true = ordre décroissant (plus grand → plus petit).
                .with_order_descending(true),
        )

        // Garde seulement les 20 meilleurs joueurs.
        .limit(20)

        // Ajoute une colonne index commençant à 1.
        .with_row_index("index", Some(1))

        // Exécute réellement la requête lazy et produit un DataFrame.
        .collect()?;
    // Calcule le temps écoulé depuis le début du programme.
    let duration = start.elapsed();

    // Affiche le temps d'exécution.
    println!("Temps d'exécution: {:?}", duration);
    // Création d'un itérateur sur les lignes du DataFrame.
    let iter = df_rows_iter!(

        // DataFrame à parcourir.
        &top_shooters,

        // Déclare le type de chaque colonne lue.
        "index" => u32,
        "fullName" => &str,
        "career_3pm" => f32

    )

    // Déclenche une erreur si l'itérateur ne peut être créé.
    .unwrap();

    // Parcourt chaque ligne du DataFrame.
    for row in iter {

        // Déstructure les valeurs de la ligne.
        let (index, full_name, career_3pm) = row.unwrap();

        // Affiche les résultats dans le terminal.
        println!("{index}. {full_name}: {career_3pm}");
    }

    
    // Retourne Ok pour indiquer que le programme s'est terminé correctement.
    Ok(())
}