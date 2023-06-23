// use image::{io::Reader as ImageReader, GenericImageView, DynamicImage};

mod nn;
use nn::{NN, Sample};

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

    let training_data = xor_data;


    let arch = [2, 2, 1];
    let mut nn = NN::new(&arch);

    // let img8 = ImageReader::open("./src/mnist/8.png").unwrap().decode().unwrap();
    // let img1 = ImageReader::open("./src/mnist/1.png").unwrap().decode().unwrap();

    // let size = img8.width() * img8.height();

    println!("All initialized!, arch: {:?}", arch);

    let mut window = Platform::new(800, 600).unwrap();
    let mut tekenen = Tekenen::new(800, 600);

    Platform::set_interval(move || {

        // Process events
        while let Some(event) = window.read_events() {
            match event {
                Event::Quit => {
                    return IntervalDecision::Stop
                },
                Event::KeyDown { char: Some(char), .. } => {
                    println!("{char}")
                },
                _ => { }
            }
        }

        // update nn
        // tick(&mut nn, &mut tekenen, &img8);

        // draw operations
        tekenen.background(COLORS::GRAY);

        // let width = img8.width();
        // let height = img8.height();

        // draw original image
        // for x in 0..width {
        //     for y in 0..height {
        //         let color = img8.get_pixel(x, y);
        //         tekenen.set_pixel(x as i32, y as i32, [color[0], color[0], color[0], 255]);
        //     }
        // }

        // draw nn image
        // for x in 0..width {
        //     for y in 0..height {
        //         let xx = x as f32 / width as f32;
        //         let yy = y as f32 / height as f32;

                // let mut color = nn.forward(&[xx, yy]) * 255.0;

                // if color < 0.0 { color = 0.0 }
                // if color > 255.0 { color = 255.0 }

                // let color = color as u8;


                // tekenen.set_pixel(x as i32 + width as i32, y as i32, [color, color, color, 255]);
            // }
        // }

        let score = format!("Score: {}", nn.score(&training_data));

        for _ in 0..100 {
            nn.train_samples(&training_data, 10.0);
        }

        tekenen.draw_text(&score, 100, 100);


        let remaining = format!("Reamining time: {:?}", Platform::get_remaining_time());
        tekenen.draw_text(&remaining, 100, 200);

        window.display_pixels(tekenen.get_pixels());

        IntervalDecision::Repeat
    }, 60)
}
