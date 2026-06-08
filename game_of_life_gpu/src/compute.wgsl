// Structure représentant une grille.
//
// Le buffer contient un tableau linéaire de u32.
// Chaque cellule vaut :
// - 0 : morte
// - 1 : vivante
struct Grid {
    cells: array<u32>,
};

// Paramètres envoyés depuis Rust.
//
// Le shader doit connaître la taille de la grille.
struct Params {
    width: u32,
    height: u32,
};

// Grille courante.
//
// Le shader lit dedans.
@group(0) @binding(0)
var<storage, read> current: Grid;

// Grille suivante.
//
// Le shader écrit dedans.
@group(0) @binding(1)
var<storage, read_write> next: Grid;

// Paramètres WIDTH / HEIGHT.
@group(0) @binding(2)
var<uniform> params: Params;

// Convertit une coordonnée 2D en index 1D.
//
// Même logique que côté Rust :
// index = y * width + x
fn index(x: u32, y: u32) -> u32 {
    return y * params.width + x;
}

// Compte les voisins vivants autour d’une cellule.
//
// On regarde les 8 cases autour de `(x, y)` :
// - haut gauche
// - haut
// - haut droite
// - gauche
// - droite
// - bas gauche
// - bas
// - bas droite
fn count_neighbors(x: u32, y: u32) -> u32 {
    var count: u32 = 0u;

    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            // On ignore la cellule elle-même.
            if (dx == 0 && dy == 0) {
                continue;
            }

            let nx = i32(x) + dx;
            let ny = i32(y) + dy;

            // On vérifie qu’on reste dans la grille.
            //
            // Ici les bords sont fermés :
            // les cellules hors grille sont simplement ignorées.
            if (
                nx >= 0 &&
                nx < i32(params.width) &&
                ny >= 0 &&
                ny < i32(params.height)
            ) {
                count = count + current.cells[index(u32(nx), u32(ny))];
            }
        }
    }

    return count;
}

// Fonction principale exécutée par le GPU.
//
// Chaque invocation correspond à une cellule de la grille.
//
// Avec `@workgroup_size(16, 16)`, le GPU travaille par blocs
// de 16 x 16 threads.
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;

    // Certains threads peuvent dépasser la taille réelle de la grille,
    // car on arrondit le nombre de workgroups vers le haut.
    //
    // Exemple :
    // WIDTH = 2000, workgroup = 16
    // 2000 n’est pas forcément multiple parfait selon les dimensions.
    if (x >= params.width || y >= params.height) {
        return;
    }

    let i = index(x, y);

    // État actuel de la cellule.
    let alive = current.cells[i] == 1u;

    // Nombre de voisins vivants.
    let neighbors = count_neighbors(x, y);

    // Règles du Game of Life :
    //
    // 1. Une cellule vivante survit avec 2 ou 3 voisins.
    // 2. Une cellule morte naît avec exactement 3 voisins.
    // 3. Sinon, la cellule meurt ou reste morte.
    if (alive && (neighbors == 2u || neighbors == 3u)) {
        next.cells[i] = 1u;
    } else if (!alive && neighbors == 3u) {
        next.cells[i] = 1u;
    } else {
        next.cells[i] = 0u;
    }
}