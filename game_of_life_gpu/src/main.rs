
use indicatif::{ProgressBar, ProgressStyle};

use rand::RngExt;
use array2d::Array2D;
use wgpu::{self, util::DeviceExt};
use image::{GrayImage, Luma};
const WIDTH:usize=1204;
const HEIGHT:usize=1204;
const ITERATIONS:i32=100000;



/// Représente une simulation du Game of Life exécutée sur le GPU.
///
/// La structure contient :
/// - les ressources GPU (device, queue),
/// - les buffers de données,
/// - le pipeline de calcul,
/// - les bind groups permettant d'alterner les buffers,
/// - les paramètres de dispatch des workgroups.
struct GameOfLife {
    /// Interface principale vers le GPU.
    ///
    /// Permet de créer les buffers, pipelines,
    /// bind groups, etc.
    device: wgpu::Device,
    /// File de commandes envoyées au GPU.
    ///
    /// Utilisée pour transférer des données
    /// et soumettre les calculs.
    queue : wgpu::Queue,
    /// Pipeline de calcul exécutant le shader
    /// du Game of Life.
    ///
    /// Contient le shader compilé ainsi que
    /// sa configuration d'exécution.
    pipeline: wgpu::ComputePipeline,
    /// Bind group utilisant :
    ///
    /// current = buffer_a
    /// next    = buffer_b
    ///
    /// Correspond généralement aux itérations paires.
    bind_ab: wgpu::BindGroup,
    /// Bind group utilisant :
    ///
    /// current = buffer_b
    /// next    = buffer_a
    ///
    /// Correspond généralement aux itérations impaires.
    bind_ba: wgpu::BindGroup,
    /// Nombre de workgroups à lancer sur l'axe X.
    ///
    /// Calculé à partir de :
    /// ceil(width / WORKGROUP_SIZE_X)
    workgroup_x: u32,
    /// Nombre de workgroups à lancer sur l'axe Y.
    ///
    /// Calculé à partir de :
    /// ceil(height / WORKGROUP_SIZE_Y)
    workgroup_y: u32,
    /// Taille totale des buffers de cellules
    /// exprimée en octets.
    ///
    /// Exemple :
    /// width * height * size_of::<u32>()
    buffer_size: wgpu::BufferAddress,
    /// Buffer temporaire utilisé pour rapatrier
    /// les données du GPU vers le CPU.
    ///
    /// Principalement utile pour :
    /// - le débogage
    /// - les tests
    /// - l'affichage côté CPU
    staging_buffer: wgpu::Buffer,
    /// Premier buffer contenant une génération.
    ///
    /// Selon l'itération, il peut servir :
    /// - de buffer de lecture (`current`)
    /// - ou de buffer d'écriture (`next`)
    buffer_a: wgpu::Buffer,

    /// Second buffer contenant une génération.
    ///
    /// Fonctionne en ping-pong avec `buffer_a`
    /// afin d'éviter de lire et écrire dans le
    /// même buffer pendant le calcul.
    buffer_b: wgpu::Buffer,
    
}

impl GameOfLife  {
    
    
    
    pub async fn init(grid:Array2D<i32>)->Self {
        // Conversion de la grille 2D en vecteur linéaire.
        //
        // Le GPU manipule un buffer 1D de mémoire.
        // Nous transformons donc la représentation logique
        // de la grille en une représentation compatible GPU.
        let mut grid_as_vec = Vec::with_capacity(WIDTH * HEIGHT);

        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let grid_value = *grid.get(y, x).unwrap_or(&0);
                grid_as_vec.push(grid_value);
            }
        }
        //Creation de l'instance du WGPU.
        let instance = wgpu::Instance::default();
        // Sélection automatique d'un adaptateur GPU.
        //
        // L'adaptateur représente la carte graphique qui sera utilisée.
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        println!("Using GPU : {:?}",adapter.get_info());
        // Création du périphérique logique (Device)
        // et de la file de commandes (Queue).
        //
        // Device :
        //   création des ressources GPU.
        //
        // Queue :
        //   envoi des commandes vers le GPU.
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("GPU Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await
            .unwrap();
        // Taille totale des buffers GPU en octets.
        let buffer_size =
            (grid_as_vec.len() * std::mem::size_of::<i32>())
                as wgpu::BufferAddress;
        // Conversion du vecteur d'entiers en tableau d'octets.
        //
        // Les buffers GPU manipulent uniquement des bytes.
        let initial_bytes: Vec<u8> = grid_as_vec
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        // Buffer A :
        //
        // Contient la génération initiale du Game of Life.
        //
        // Il servira alternativement de buffer de lecture
        // ou d'écriture selon l'itération.
        let buffer_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid A"),
            contents: &initial_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        }); 
        // Buffer B :
        //
        // Buffer secondaire utilisé pour le ping-pong buffering.
        //
        // Pendant qu'un buffer est lu par le shader,
        // l'autre reçoit le résultat.    
        let buffer_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Grid B"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Buffer B :
        //
        // Buffer secondaire utilisé pour le ping-pong buffering.
        //
        // Pendant qu'un buffer est lu par le shader,
        // l'autre reçoit le résultat.
        let staging_buffer =
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Readback Buffer"),
                size: buffer_size,
                usage: wgpu::BufferUsages::MAP_READ
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        // Chargement et compilation du shader WGSL.
        //
        // Ce shader contient l'algorithme du Game of Life
        // exécuté sur le GPU.
        let shader =
            device.create_shader_module(
                wgpu::ShaderModuleDescriptor {
                    label: Some("Game of Life Shader"),
                    source: wgpu::ShaderSource::Wgsl(
                        include_str!("compute.wgsl").into(),
                    ),
                },
            );
        // Définition des ressources accessibles par le shader.
        //
        // Binding 0 : buffer de lecture.
        // Binding 1 : buffer d'écriture.
        // Binding 2 : paramètres de simulation.
        let bind_group_layout =
        device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage {
                                read_only: true,
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage {
                                read_only: false,
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            },
        );
        // Description de l'organisation globale du pipeline.
        let pipeline_layout =
            device.create_pipeline_layout(
                &wgpu::PipelineLayoutDescriptor {
                    label: Some("Pipeline Layout"),
                    bind_group_layouts: &[Some(
                        &bind_group_layout,
                    )],
                    immediate_size: 0,
                },
            );
        // Création du pipeline de calcul.
        //
        // Le pipeline représente le programme GPU prêt à être exécuté.
        let pipeline =
            device.create_compute_pipeline(
                &wgpu::ComputePipelineDescriptor {
                    label: Some("Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                },
            );
        // Paramètres envoyés au shader.
        //
        // Le shader doit connaître la largeur et la hauteur de la grille.
        let params = [WIDTH as u32, HEIGHT as u32];

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params Buffer"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });    
        // Bind group A -> B.
        //
        // Lecture dans A, écriture dans B.
        let bind_ab = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("A -> B"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Bind group B -> A.
        //
        // Lecture dans B, écriture dans A.
        //
        // Cela permet d’alterner les buffers à chaque itération sans recopier les données.
        let bind_ba = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("B -> A"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        // Taille des workgroups définie dans le shader.
        //
        // @workgroup_size(16,16)
        let workgroup_size_x = 16;
        let workgroup_size_y = 16;
        // Nombre total de workgroups nécessaires
        // pour couvrir toute la grille.
        //
        // L'arrondi supérieur garantit que
        // toutes les cellules sont traitées.
        let workgroup_x = (WIDTH as u32 + workgroup_size_x - 1) / workgroup_size_x;
        let workgroup_y = (HEIGHT as u32 + workgroup_size_y - 1) / workgroup_size_y;
        Self {
            
            device,
            queue,
            pipeline,
            bind_ab,
            bind_ba,
            workgroup_x,
            workgroup_y,
            buffer_size,
            staging_buffer,
            buffer_a,
            buffer_b,
        }
    }
    /// Exécute une génération du Game of Life sur le GPU.
    ///
    /// Cette fonction :
    /// 1. prépare une commande GPU,
    /// 2. sélectionne le bon couple de buffers,
    /// 3. lance le shader sur toute la grille,
    /// 4. envoie le travail au GPU.
    pub fn iterate (&self,i:i32) {
        // Création d'un encodeur de commandes.
        //
        // Les commandes GPU ne sont pas exécutées immédiatement.
        // Elles sont d'abord enregistrées dans un CommandEncoder.
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Encoder"),
        });

        {
            // Création d'un encodeur de commandes.
            //
            // Les commandes GPU ne sont pas exécutées immédiatement.
            // Elles sont d'abord enregistrées dans un CommandEncoder.
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            // Sélection du pipeline contenant
            // le shader WGSL du Game of Life.
            pass.set_pipeline(&self.pipeline);
            // Sélection du bind group.
            //
            // Les itérations paires utilisent :
            //
            // current = buffer_a
            // next    = buffer_b
            //
            // Les itérations impaires utilisent :
            //
            // current = buffer_b
            // next    = buffer_a
            //
            // Cette technique est appelée
            // "ping-pong buffering".
            if i % 2 == 0 {
                pass.set_bind_group(0, &self.bind_ab, &[]);
            } else {
                pass.set_bind_group(0, &self.bind_ba, &[]);
            }
            // Lancement du shader.
            //
            // Le GPU crée :
            //
            // workgroup_x × workgroup_y
            //
            // groupes de travail.
            //
            // Chaque workgroup contient
            // 16 × 16 threads (défini dans le shader).
            pass.dispatch_workgroups(self.workgroup_x, self.workgroup_y, 1);
        }
        // Finalisation de la liste de commandes
        // puis envoi au GPU.
        //
        // L'exécution est asynchrone :
        // le CPU continue immédiatement son travail
        // pendant que le GPU calcule la génération.
    
        self.queue.submit(Some(encoder.finish()));
    }
    pub fn get_grid (&self, i:i32)->Array2D<i32> {
        // Choix du buffer final.
        //
        // Si le nombre d’itérations est pair, le résultat est dans A.
        // Sinon, il est dans B.
        let final_buffer = if i % 2 == 0 {
            &self.buffer_a
        } else {
            &self.buffer_b
        };

        // Copie du buffer final GPU vers le staging buffer.
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Readback Encoder"),
        });

        encoder.copy_buffer_to_buffer(final_buffer, 0, &self.staging_buffer, 0, self.buffer_size);

        self.queue.submit(Some(encoder.finish()));

        // Mapping du staging buffer pour lecture CPU.
        let slice = self.staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();

        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        // On attend que le GPU ait fini la copie.
        

        // Force le device à traiter les commandes GPU
        // et attendre la fin du copy + mapping.
        self.device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .unwrap();

        

        receiver.recv().unwrap().unwrap();

        // Lecture des données finales.
        let data = slice.get_mapped_range();
        // Conversion &[u8] -> &[u32].
        let cells_u32: &[u32] = bytemuck::cast_slice(&data);

        // Conversion Vec<u32> -> Vec<i32>.
        let cells_i32: Vec<i32> = cells_u32
            .iter()
            .map(|&v| v as i32)
            .collect();

        // Important : libérer la vue mappée avant unmap().
        drop(data);

        self.staging_buffer.unmap();

        // Reconstruction de la grille 2D.
        //
        // Adapte ici selon les noms de tes champs :
        // self.width / self.height ou constantes WIDTH / HEIGHT.
        Array2D::from_row_major(
            &cells_i32,
            HEIGHT as usize,
            WIDTH as usize,
        ).unwrap()
    }
}
pub fn save_array2d_as_luma_image(
    grid: &Array2D<i32>,
    path: &str,
) -> Result<(), image::ImageError> {
    let height = grid.num_rows();
    let width = grid.num_columns();

    let mut img = GrayImage::new(width as u32, height as u32);

    for y in 0..height {
        for x in 0..width {
            let value = grid[(y, x)];

            // 0 = noir, 1 = blanc
            let pixel_value = if value == 1 { 255u8 } else { 0u8 };

            img.put_pixel(
                x as u32,
                y as u32,
                Luma([pixel_value]),
            );
        }
    }

    img.save(path)
}
fn main() {
    //la barre de progression pour voir l'avancement sur la ligne de commande.
    let pb = ProgressBar::new(ITERATIONS.try_into().unwrap()); 
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} {percent}%"
        )
        .unwrap()
        .progress_chars("#>-"),
    );
    //on va initialiser la grille avec des valeurs random.
    let mut rng = rand::rng();
    let values: Vec<i32> = (0..WIDTH * HEIGHT)
        .map(|_| rng.random_bool(0.25) as i32)
        .collect();

    let initial_grid = Array2D::from_row_major(
        &values,
        HEIGHT,
        WIDTH,
    ).unwrap();
    //on sauvegarde l'image de depart.
    save_array2d_as_luma_image(&initial_grid, "gpu_start.png").unwrap();
    //initialisation du jeu de la vie.
    //utilisation de pollster, car la création des ressources sur le GPU est asyncrone.
    //on attend que les ressources soient pretes.
    let game_of_life = pollster::block_on(GameOfLife::init(initial_grid));

    for i in 0..ITERATIONS {
        game_of_life.iterate(i);
        
        pb.inc(1);
    }
    //on récupere les données de la VRAM vers la RAM au travers du staging buffer.
    let final_grid = game_of_life.get_grid(ITERATIONS);
    //sauvegarde de la grille finale sou la forme d'une image en noir et blanc.

    save_array2d_as_luma_image(&final_grid, "gpu_result.png").unwrap();
    
}