#![allow(unused_variables)]

mod basic;
mod preloaded;

use rand::seq::SliceRandom;
use tekenen::*;
use platform::*;

mod nn;
use nn::{Input, Sample};

use std::{time::Instant};

use crate::nn::Output;

use image::GenericImageView;

use serde_json;
use std::fs;

fn load_data(data: &Tekenen, number: usize) -> Sample {
    let size = data.width() * data.height();

    let layer = Input::new(size);
    let pixels = data.get_pixels();

    {
        let mut bias = layer.bias_b.borrow_mut();
        let mut value = layer.value_a.borrow_mut();
        for i in 0..size {
            bias[i] = 0.0;
            value[i] = pixels[i * 4] as f32 / 255.0;
        };
    }

    let mut output = Vec::new();

    for i in 0..=9 {
        if i == number {
            output.push(1.0);
        } else {
            output.push(0.0);
        }
    }

    Sample { input: layer, output }
}

fn heighest(out: &Vec<f32>) -> usize {
    let mut record = -1.0;
    let mut holder = 0;

    for i in 0..=9 {
        if out[i] > record {
            record = out[i];
            holder = i;
        }
    }

    holder
}

fn score_all(nn: &mut Box<nn::NN>, data: &Vec<Sample>) -> String {
    let mut right = 0;

    for data in data.iter() {
        let out = nn.get(&data.input);

        if heighest(&out) == heighest(&data.output) {
            right += 1;
        }
    }

    format!("{right}/{}, {}%", data.len(), right as f32 / data.len() as f32 * 100.0)
}

fn load_set(path: &str) -> Vec<Sample> {
    let mut samples = Vec::new();

    for i in 0..=9 {
        let mut current = 0;

        for file in std::fs::read_dir(std::path::Path::new(&format!("{path}/{i}"))).unwrap() {
            let file = file.unwrap();

            let img = image::io::Reader::open(file.path().to_str().unwrap()).unwrap().decode().unwrap();

            let mut vec = vec![];

            for y in 0..img.height() {
                for x in 0..img.width() {
                    let color = img.get_pixel(x, y);
                    vec.push(color[0]);
                    vec.push(color[1]);
                    vec.push(color[2]);
                    vec.push(color[3]);
                }
            };

            let width = img.width();
            let height = img.height();

            let tekenen = Tekenen::from_pixels(width as usize, height as usize, vec);

            samples.push(load_data(&tekenen, i));

            // if current > 900 {
            //     break
            // }

            current += 1;
        }

        println!("Read {current} files in: {path}/{i}");
    }

    samples
}

fn main () {
    let args: Vec<String> = std::env::args().collect();

    let basic = args.iter().any(|el: &String| { ["-b", "-basic", "--basic"].contains(&el.as_str()) });
    let preload = args.iter().any(|el: &String| { ["-p", "-preload", "--preload", "-preloaded", "--preloaded"].contains(&el.as_str()) });
    let help = args.iter().any(|el: &String| { ["-h", "-help", "--help"].contains(&el.as_str()) });

    if help {
        println!("Usage of nn:
        
        <-h, --help>    Show this message
        <-b, --basic>   Load basic variant, memorize the numbers 1-10
        <-p, --preload> Use prelaoded data baked into the binary");
        return;
    }

    if basic {
        basic::basic();
        return;
    }

    // load images
    let preloaded = preloaded::load_preloaded();

    let (mut training_data, testing_data) = if preload {
        let mut training_data = Vec::new();
        let mut testing_data = Vec::new();

        for (i, imgs) in preloaded.training.iter().enumerate() {
            for img in imgs.iter() {
                training_data.push(load_data(img, i))
            }
        }

        for (i, imgs) in preloaded.testing.iter().enumerate() {
            for img in imgs.iter() {
                testing_data.push(load_data(img, i))
            }
        }

        (training_data, testing_data)
    } else {
        let training = load_set("./src/mnist/training");
        let testing = load_set("./src/mnist/testing");
        
        (training, testing)
    };

    let mut window = Platform::new(800, 600).unwrap();
    let mut tekenen = Tekenen::new(800, 600);

    let mut showing_i = 0;

    let arch = [28*28, 32, 16, 10];
    let mut nn = nn::NN::new(&arch);

    let mut parameters = 0;
    for i in 1..arch.len() {
        parameters += arch[i];
        parameters += arch[i] * arch[i - 1];
    }

    let mut running = true;
    let mut training_iterations = 0;
    let mut started = Instant::now();

    let mut batch_slider = ui::widgets::Slider::new_sized(0, 200, 50, 1.0, 200.0, 20.0);
    let mut testing = 0;
    let mut learning_rate = 0.1;

    let mut graph_data = Vec::new();

    let mut correct = "Press <c> to update".to_owned();

    let mut showing_map = false;
    let mut drawing = false;
    let mut mouse_down = None;
    let mut drawing_canvas = Tekenen::new(280, 280);
    let mut drawing_sample_canvas = Tekenen::new(28, 28);
    let mut drawing_sample = load_data(&drawing_sample_canvas, 0);

    let mut shuffled_until = usize::MAX;

    for i in 0..=9 {
        drawing_sample.output[i] = i as f32;
    }

    Platform::set_interval(move || {
        // Process events
        while let Some(event) = window.read_events() {
            match event {
                Event::Quit => {
                    return IntervalDecision::Stop
                },
                Event::KeyDown { char: Some(char), .. } => {
                    match char {
                        ' ' => running = !running,
                        'i' => learning_rate *= 2.0,
                        'o' => learning_rate /= 2.0,
                        'n' => testing += testing_data.len() / 10 + 1,
                        'm' => testing -= testing_data.len() / 10 + 1,
                        'c' => correct = score_all(&mut nn, &testing_data),
                        's' => showing_map = !showing_map,
                        'l' => {
                            let data = fs::read_to_string("./saved_nn.json").unwrap();
                            nn = serde_json::from_str(&data).unwrap();
                        },
                        'k' => {
                            let data = serde_json::to_string(&nn).unwrap();
                            fs::write("./saved_nn.json", data).unwrap();
                        },
                        'd' => {
                            drawing = !drawing;
                            drawing_canvas.background([0, 0, 0, 255]);
                        },
                        'r' => {
                            started = Instant::now();
                            training_iterations = 0;
                            nn = nn::NN::new(&arch);
                            graph_data = Vec::new();
                        },
                        _ => { }
                    }

                    println!("{char}")
                },
                Event::MouseDown { x, y } => {
                    batch_slider.mouse_down(x, y);
                    mouse_down = Some((x, y));
                },
                Event::MouseMove { x, y } => {
                    batch_slider.mouse_move(x, y);
                    if let Some(_) = mouse_down {
                        mouse_down = Some((x, y));
                    }
                },
                Event::MouseUp { x, y } => {
                    batch_slider.mouse_up(x, y);
                    mouse_down = None;
                },
                _ => { }
            }
        }

        // Train AI
        let mut rng = rand::thread_rng();
        let running_time = Instant::now() - started;

        let batch_size = batch_slider.value as usize;
        'out: while !Platform::get_remaining_time().is_zero() && running {
            if shuffled_until >= training_data.len() - batch_size {
                shuffled_until = 0;
                training_data.shuffle(&mut rng);
            }

            while shuffled_until + batch_size < training_data.len() {
                let start = shuffled_until;
                let end = start + batch_size;
    
                nn.train_samples(&training_data[start..end], learning_rate);

                training_iterations += batch_size;

                if Platform::get_remaining_time().is_zero() {
                    break 'out
                }
            }
        }

        // Draw
        tekenen.background(colors::GRAY);
        batch_slider.display(&mut tekenen);

        // Print info
        let score = nn.score(&testing_data[0..200]);
        graph_data.push(score);

        let infos = [
            format!("Score: {score}"),
            format!("Test: {correct}"),
            format!("Learning rate: {learning_rate}"),
            format!("Batch size: {batch_size}"),
            format!("Iteration: {training_iterations}"),
            format!("Elapsed: {}", running_time.as_secs()),
            format!("Arch: {:?}", arch),
            format!("Training size: {:?}", training_data.len()),
            format!("Testing size: {:?}", testing_data.len()),
            format!("Paramters: {parameters}"),
            "".to_string(),
            "< >: Pause/Unpause".to_string(),
            "<c>: Test all images".to_string(),
            "<i>: Double trainig rate".to_string(),
            "<o>: Half training rate".to_string(),
            "<n>: Show next test".to_string(),
            "<m>: Show previous test".to_string(),
            "<d>: Draw".to_string(),
            "<s>: Show neuron map".to_string(),
            "<l>: Load AI".to_string(),
            "<k>: Save AI".to_string(),
        ];

        for (i, info) in infos.iter().enumerate() {
            tekenen.draw_text(info, 25, 75 + 25 * i as i32);
        }

        // Draw top images
        for i in 0..preloaded.testing.len() {
            let image = &preloaded.testing[i][showing_i % &preloaded.testing[i].len()];
            tekenen.draw_image(29 * i as i32, 0, image)
        }
        showing_i += 1;

        let x1 = 500;
        let y1 = 20;
        let scale: i32 = 10;
        let size = scale * 28;

        if drawing {
            if let Some((x, y)) = mouse_down {
                drawing_canvas.circle(x - x1, y - y1, 15, [255, 255, 255, 255]);

                for x in 0..28 {
                    for y in 0..28 {
                        let mut total = 0;

                        for dx in 0..scale {
                            for dy in 0..scale {
                                total += drawing_canvas.get_pixel(x * scale + dx, y * scale + dy).unwrap()[0] as i32
                            } 
                        }

                        let c = (total / (scale * scale)) as u8;
                        drawing_sample_canvas.set_pixel(x, y, [c, c, c, 255])
                    } 
                }

                drawing_sample = load_data(&drawing_sample_canvas, 0);
                for i in 0..=9 {
                    drawing_sample.output[i] = i as f32;
                }
            }

            tekenen.draw_image(x1, y1, &drawing_canvas);
        } else if showing_map {
            let index_1 = testing % nn.layers[1].value_a.borrow().len();
            let connections = &nn.connections[0];

            for x in 0..28i32 {
                for y in 0..28i32 {
                    let index_2 = y * 28 + x;
                    let mut c = connections.value_w[(index_1, index_2 as usize)];

                    c = c / 2.0 + 0.5;
                    if c < 0.0 { c = 0.0 }
                    if c > 1.0 { c = 1.0 }

                    let c = (c * 255.0) as u8;
                    let x = x1 + x * scale;
                    let y = y1 + y * scale;

                    tekenen.rect(x, y, scale, scale, [255 - c, c, 0, 255]);
                }
            }
        } else {
            // Display graph
            let max_points = 100;
            let step = graph_data.len() / max_points + 1;
            let count = graph_data.len() / step;

            tekenen.rect(x1, y1, size, size, colors::WHITE);

            let max_h = graph_data[0];
            let mut px = x1;
            let mut py = y1;
            
            for i in 0..count {
                let h = graph_data[i * step];

                let y = y1 + size - (size as f32 * h / max_h) as i32;

                tekenen.line(px, py, px + size / count as i32, y, colors::RED);
                px += size / count as i32;
                py = y;
            }
        }

        let testing_img = if drawing {
            &drawing_sample
        } else {
            &testing_data[testing % testing_data.len()]
        };


        let img_data = testing_img.input.value_a.borrow();
        let result = nn.get(&testing_img.input);

        let x1 = 400;
        let y1 = 350;

        for dx in 0..28 {
            for dy in 0..28 {
                let i = dy * 28 + dx;
                let x = x1 + dx;
                let y = y1 + dy;

                let mut c = img_data[i];
                if c < 0.0 { c = 0.0 }
                if c > 1.0 { c = 1.0 }

                let c = (c * 255.0) as u8;

                tekenen.set_pixel(x as i32, y as i32, [c, c, c, 255]);
            }
        }

        let mut dispay = |out: &Output, x: i32, y: i32| {
            let holder = heighest(out);
            
            for i in 0..=9 {
                if i == holder {
                    tekenen.draw_text(&format!("[{}]", out[i]), x, y + i as i32 * 25);
                } else {
                    tekenen.draw_text(&format!(" {}", out[i]), x, y + i as i32 * 25);
                }
            }
        };

        dispay(&testing_img.output, x1 as i32 + 50, y1 as i32);
        dispay(&result, x1 as i32 + 125, y1 as i32);

        window.display_pixels(tekenen.get_pixels());
        
        IntervalDecision::Repeat
    }, 10)
}