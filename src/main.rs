use std::{arch::x86_64::_CMP_TRUE_UQ, thread::current};

use rand::Rng;
mod tekenen;
mod sdl;
use sdl::Event;
use tekenen::Tekenen;

use image::{io::Reader as ImageReader, GenericImageView, DynamicImage};

#[derive(Debug)]
struct Matrix {
    weights: Vec<f32>,
    biases: Vec<f32>
}

impl Matrix {
    fn new(current: u32, previous: u32) -> Self {
        let size = current as usize * previous as usize;
        let mut weights = Vec::with_capacity(size);

        let mut rng = rand::thread_rng();
        for _ in 0..size {
            weights.push(rng.gen_range(-1.0..1.0))
        }

        let mut biases = Vec::with_capacity(current as usize);
        for _ in 0..current {
            biases.push(rng.gen_range(-1.0..1.0))
        }

        Self {
            weights,
            biases
        }
    }
}

#[derive(Debug)]
struct NN {
    layers: Vec<Vec<f32>>,
    matrices: Vec<Matrix>,
    gradient: Vec<Matrix>,
    expected: Vec<Vec<f32>>,
}

impl NN {
    fn new(arch: &[u32]) -> Box<Self> {
        assert!(arch.len() > 1, "Invalid arch");

        let mut layers = Vec::with_capacity(arch.len());
        for layer in 0..arch.len() {
            layers.push(vec![0.0; arch[layer] as usize])
        }

        let mut expected = Vec::with_capacity(arch.len());
        for layer in 0..arch.len() {
            expected.push(vec![0.0; arch[layer] as usize])
        }

        // layer 0 (input layer) has no matrix
        let mut matrices = Vec::with_capacity(arch.len() - 1);
        for matrix in 1..arch.len() {
            matrices.push(Matrix::new(arch[matrix], arch[matrix - 1]))
        }

        // layer 0 (input layer) has no matrix
        let mut backprop = Vec::with_capacity(arch.len() - 1);
        for matrix in 1..arch.len() {
            backprop.push(Matrix::new(arch[matrix], arch[matrix - 1]))
        }

        Box::new(Self {
            layers,
            matrices,
            gradient: backprop,
            expected
        })
    }

    fn forward(&mut self, input: &[f32]) -> f32 {
        // let out = io[0] as f32 * nn.wa + io[1] as f32 * nn.wb + nn.b;

        // layer 0 is input
        assert_eq!(input.len(), self.layers[0].len());

        // could probably swap layer 1 for input layer?
        for i in 0..input.len() {
            self.layers[0][i] = input[i]
        }

        for layer in 1..self.layers.len() {
            let matrix = &self.matrices[layer - 1];


            for curr_node in 0..self.layers[layer].len() {
                let prev = &self.layers[layer - 1];


                // add the bias
                let mut out = matrix.biases[curr_node];
        
                // add all weights
                for prev_node in 0..prev.len() {
                    out += prev[prev_node] * matrix.weights[curr_node * prev.len() + prev_node]
                }
        
                // activation function
                self.layers[layer][curr_node] = activate_sigmoid(out)
            }
        }

        self.layers[self.layers.len() - 1][0]
    }
}

// const DATA_AND: [(Input, Output); 4] = [
//     ([0, 0], 0.0),
//     ([0, 1], 0.0),
//     ([1, 0], 0.0),
//     ([1, 1], 1.0),
// ];

// const DATA_OR: [(Input, Output); 4] = [
//     ([0, 0], 0.0),
//     ([0, 1], 1.0),
//     ([1, 0], 1.0),
//     ([1, 1], 1.0),
// ];

// const DATA_XOR: [(Input, Output); 4] = [
//     ([0, 0], 0.0),
//     ([0, 1], 1.0),
//     ([1, 0], 1.0),
//     ([1, 1], 0.0),
// ];

// const DATA: [(Input, Output); 4] = DATA_XOR;

fn activate_linear(value: f32) -> f32 {
    if value > 0.0 {
        value
    } else {
        0.0
    }
}

fn activate_sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + f32::powf(std::f32::consts::E, -value))
}



// fn score(nn: &mut NN, silent: bool) -> f32 {
//     let mut total = 0.0;

//     DATA.into_iter().for_each(|el| {
//         for i in 0..el.0.len() {
//             nn.layers[0][i] = el.0[i] as f32;
//         }

//         run(nn);
//         let got = nn.layers[nn.layers.len() - 1][0];
        
//         let err = got - el.1;

//         if !silent {
//             println!("Run with: {:?}, expected: {} got: {}", el.0, el.1, got);
//         }

//         total += err * err;
//     });

//     total / DATA.len() as f32
// }

fn score_img(nn: &mut NN, img: &DynamicImage) -> f32 {
    let mut total = 0.0;

    let width = img.width();
    let height = img.height();
    let size = width * height;

    for x in 0..width {
        for y in 0..height {
            let xx = x as f32 / width as f32;
            let yy = y as f32 / height as f32;

            let got = nn.forward(&[xx, yy]);
        
            let err = got - img.get_pixel(x, y)[0] as f32;
            total += err * err;
        }
    }

    total / size as f32
}

const EPSILON: f32 = 0.1;
const RATE: f32 = 0.1;

fn gym(nn: &mut NN, img: &DynamicImage) {
    let start = score_img(nn, img);

    // calculate gradient
    for i in 0..nn.matrices.len() {
        for weight in 0..nn.matrices[i].weights.len() {
            nn.matrices[i].weights[weight] += EPSILON;

            let delta = score_img(nn, img) - start;
            nn.gradient[i].weights[weight] = delta / EPSILON;

            nn.matrices[i].weights[weight] -= EPSILON;
        }
    
        for bias in 0..nn.matrices[i].biases.len() {
            nn.matrices[i].biases[bias] += EPSILON;

            let delta = score_img(nn, img) - start;
            nn.gradient[i].biases[bias] = delta / EPSILON;

            nn.matrices[i].biases[bias] -= EPSILON;
        }
    }

    // Apply gradient
    for i in 0..nn.matrices.len() {        
        for weight in 0..nn.matrices[i].weights.len() {
            nn.matrices[i].weights[weight] -= nn.gradient[i].weights[weight] * RATE;
        }
    
        for bias in 0..nn.matrices[i].biases.len() {
            nn.matrices[i].biases[bias] -= nn.gradient[i].biases[bias] * RATE;
        }
    }
}

fn gym_backprop(nn: &mut NN, img: &DynamicImage) {
    let start = score_img(nn, img);

    let width = img.width();
    let height = img.height();
    // calculate gradient
    for x in 0..width {
        for y in 0..height {
            let xx = x as f32 / width as f32;
            let yy = y as f32 / height as f32;

            nn.forward(&[xx, yy]);

            // store the expected value
            let last = nn.expected.len() - 1;
            nn.expected[last][0] = img.get_pixel(x, y)[0] as f32;

            let mut curr_layer = nn.expected.len() - 1;
            while curr_layer > 0 {
                for node in 0..nn.layers.len() {
                    let current = nn.layers[curr_layer][node];
                    let expected = nn.expected[curr_layer][node];

                    // set activation of previous layer
                    nn.gradient[curr_layer - 1].biases[node] = 2.0 * current * expected * (1.0 - current);

                    for prev_node in 0..nn.layers[curr_layer - 1].len() {

                    }
                }

                curr_layer -= 1;
            }
        }
    }

    // Apply gradient
    for i in 0..nn.matrices.len() {        
        for weight in 0..nn.matrices[i].weights.len() {
            nn.matrices[i].weights[weight] -= nn.gradient[i].weights[weight] * RATE;
        }
    
        for bias in 0..nn.matrices[i].biases.len() {
            nn.matrices[i].biases[bias] -= nn.gradient[i].biases[bias] * RATE;
        }
    }
}

fn tick(nn: &mut Box<NN>, tekenen: &mut Tekenen, img: &DynamicImage) {

    // println!("{:?}", nn);
    // println!("Initial score: {}", score(nn, false));

    // for _ in 0..20_000{
    //     gym(nn);
    // }

    // println!("{:?}", nn);
    // println!("Intermidiate score: {}", score(nn, true));

    for _ in 0..1{
        gym_backprop(nn, img);
    }

    // println!("{:?}", nn);
    // println!();
    // let nn_text = format!("{:?}", nn);
    let text = format!("Final score: {}", score_img(nn, img));
    tekenen.draw_text(&text, 50, 50);
}

fn main() {
    let arch = [2, 5, 5, 1];
    let mut nn = NN::new(&arch);

    let mut platform = sdl::SDLPlatform::new(800, 600);

    let mut tekenen = tekenen::Tekenen::new(800, 600);

    tekenen.background(tekenen::BLACK);

    let img8 = ImageReader::open("./src/mnist/8.png").unwrap().decode().unwrap();
    let img1 = ImageReader::open("./src/mnist/1.png").unwrap().decode().unwrap();

    let size = img8.width() * img8.height();

    println!("All initialized!, {size}");
    sdl::SDLPlatform::set_interval(Box::new(move || {
        while let Some(event) = platform.read_events() {
            match event {
                Event::Quit => {
                    // true indicates to interrupt the loop
                    return true;
                }
                Event::KeyDown { char, keycode, .. } => {
                    if let Some(c) = char {
                        println!("char {:?}", c)
                    } else {
                        println!("unknown char {:?}", keycode)
                    }
                }
            }
        }

        // self.terminal.render(tekenen, 0);
        tekenen.background([51, 51, 51, 255]);

        tick(&mut nn, &mut tekenen, &img8);

        let width = img8.width();
        let height = img8.height();

        for x in 0..width {
            for y in 0..height {
                let color = img8.get_pixel(x, y);
                tekenen.set_pixel(x as i32, y as i32, [color[0], color[0], color[0], 255]);
            }
        }

        for x in 0..width {
            for y in 0..height {
                let xx = x as f32 / width as f32;
                let yy = y as f32 / height as f32;

                let mut color = nn.forward(&[xx, yy]) * 255.0;

                if color < 0.0 { color = 0.0 }
                if color > 255.0 { color = 255.0 }

                let color = color as u8;


                tekenen.set_pixel(x as i32 + width as i32, y as i32, [color, color, color, 255]);
            }
        }

        platform.display_pixels(tekenen.get_pixels());

        // should not stop
        false
    }), 60);
}
