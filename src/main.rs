use image::{io::Reader as ImageReader, GenericImageView};

fn read_image(path: &str) -> Samples {
    let img = ImageReader::open(path).unwrap().decode().unwrap();

    assert_eq!(img.width(), 28);
    assert_eq!(img.height(), 28);

    let mut samples = Vec::with_capacity(28*28);

    for x in 0..28 {
        for y in 0..28 {
            let color = img.get_pixel(x, y);
            samples.push(Sample { input: vec![x as f32 / 28.0, y as f32 / 28.0], output: vec![color[0] as f32 / 255.0] })
        }
    };

    samples
}

////////////////////////////////////////////////////////////////////////////////////////////////////

mod nn;
use nn::{NN, Sample, Samples};

use rand::seq::SliceRandom;
use tekenen::{Platform, PlatformTrait, IntervalDecision, Event, Tekenen, COLORS};

fn main() {
    let and_data = vec![
        Sample { input: vec![0.0, 0.0], output: vec![0.0] },
        Sample { input: vec![0.0, 1.0], output: vec![0.0] },
        Sample { input: vec![1.0, 0.0], output: vec![0.0] },
        Sample { input: vec![1.0, 1.0], output: vec![1.0] },
    ];

    let or_data = vec![
        Sample { input: vec![0.0, 0.0], output: vec![0.0] },
        Sample { input: vec![0.0, 1.0], output: vec![1.0] },
        Sample { input: vec![1.0, 0.0], output: vec![1.0] },
        Sample { input: vec![1.0, 1.0], output: vec![1.0] },
    ];

    let xor_data = vec![
        Sample { input: vec![0.0, 0.0], output: vec![0.0] },
        Sample { input: vec![0.0, 1.0], output: vec![1.0] },
        Sample { input: vec![1.0, 0.0], output: vec![1.0] },
        Sample { input: vec![1.0, 1.0], output: vec![0.0] },
    ];

    let arch = [2, 8, 8, 8, 8, 1];
    let mut nn = NN::new(&arch);

    let img8 = read_image("./src/mnist/8.png");
    let img1 = read_image("./src/mnist/1.png");

    let mut training_data = img8;
    let mut training_iterations = 0;


    println!("All initialized!, arch: {:?}", arch);

    let mut window = Platform::new(800, 600).unwrap();
    let mut tekenen = Tekenen::new(800, 600);

    let mut rate = 0.1;
    let mut graph = vec![];

    Platform::set_interval(move || {

        // Process events
        while let Some(event) = window.read_events() {
            match event {
                Event::Quit => {
                    return IntervalDecision::Stop
                },
                Event::KeyDown { char: Some(char), .. } => {
                    match char {
                        'i' => rate *= 2.0,
                        'o' => rate /= 2.0,
                        'p' => println!("{:?}", nn),
                        _ => { }
                    }
                    println!("{char}")
                },
                _ => { }
            }
        }

        // draw operations
        tekenen.background(COLORS::GRAY);

        let score = nn.score(&training_data);
        graph.push(score);
        tekenen.draw_text(&format!("Score: {}", score), 450, 450);
        tekenen.draw_text(&format!("Iteration: {}", training_iterations), 450, 475);
        tekenen.draw_text(&format!("Rate: {rate}"), 450, 500);


        let mut rng = rand::thread_rng();

        let batch_size = 10;
        while !Platform::get_remaining_time().is_zero() {
            training_data.shuffle(&mut rng);

            for i in 0..(training_data.len() / batch_size) - 1 {
                let start = i * batch_size;
                let end = start + batch_size;
    
                nn.train_samples(&training_data[start..end], rate);
            }

            training_iterations += 1;
        }

        // draw original image
        let scale = 5;
        for sample in &training_data {
            let x = sample.input[0] * 28.0;
            let y = sample.input[1] * 28.0;
            let mut c = sample.output[0];

            if c < 0.0 { c = 0.0 }
            if c > 1.0 { c = 1.0 }

            let c = (c * 255.0) as u8;

            tekenen.rect(x as i32 * scale, y as i32 * scale, scale, scale, [c, c, c, 255]);
        }

        // draw nn image
        let size = 28 * scale;
        for x in 0..size {
            for y in 0..size {
                let mut c = nn.get(&vec![x as f32 / size as f32, y as f32 / size as f32])[0];

                if c < 0.0 { c = 0.0 }
                if c > 1.0 { c = 1.0 }

                let c = (c * 255.0) as u8;

                tekenen.set_pixel(x as i32 + scale * 29, y as i32, [c, c, c, 255]);
            }
        }

        for layer_i in 0..nn.layers.len() {
            let layer = &nn.layers[layer_i];

            for neuron_i in 0..layer.len() {
                let mut c = layer[neuron_i].bias_b;
                let mut b = 0;
                if c < -1.0 { c = -1.0; b = 100 }
                if c > 1.0 { c = 1.0; b = 100 }

                let c = ((c + 1.0) / 2.0 * 255.0) as u8;

                tekenen.circle(layer_i as i32 * 100 + 50, 200 + neuron_i as i32 * 20, 20, [255 - c, c, b, 255])
            }
        }

        for layer_i in 1..nn.layers.len() {
            let connection = &nn.connections[layer_i - 1];
            let curr_layer = &nn.layers[layer_i];
            let prev_layer = &nn.layers[layer_i - 1];

            for curr_i in 0..curr_layer.len() {
                for prev_i in 0..prev_layer.len() {
                    let mut c = connection[curr_i][prev_i].value_w;
                    let mut b = 0;
                    if c < -1.0 { c = -1.0; b = 100 }
                    if c > 1.0 { c = 1.0; b = 100 }
    
                    let c = ((c + 1.0) / 2.0 * 255.0) as u8;

                    let x1 = (layer_i - 1) as i32 * 100 + 50;
                    let x2 = layer_i as i32 * 100 + 50;

                    let y1: i32 = 200 + prev_i as i32 * 20;
                    let y2 = 200 + curr_i as i32 * 20;
    
                    tekenen.line(x1, y1, x2, y2, [255 - c, c, b, 255])
                }
            }
        }

        // draw graph
        let x1 = 450;
        let y1 = 50;
        let size = 300;
        tekenen.rect(x1, y1, size, size, COLORS::WHITE);

        let step = size as f32 / graph.len() as f32;
        let mut x = x1 as f32;
        let mut px = x1;
        let mut py = y1;

        for i in 0..graph.len() {
            let y = y1 + size - (graph[i] * size as f32) as i32;

            x += step;
            tekenen.line(px, py, x as i32, y, COLORS::RED);
            px = x as i32;
            py = y;
        }

        let remaining = format!("Reamining time: {:?}", Platform::get_remaining_time());
        tekenen.draw_text(&remaining, 450, 525);

        window.display_pixels(tekenen.get_pixels());

        IntervalDecision::Repeat
    }, 10)
}
