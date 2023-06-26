mod preloaded;

fn tekenen_to_sample(image: &Tekenen, value: f32) -> Samples {
    assert_eq!(image.width(), 28);
    assert_eq!(image.height(), 28);

    let mut samples = Vec::with_capacity(28*28);

    for x in 0..28 {
        for y in 0..28 {
            let color = image.get_pixel(x, y).unwrap();
            samples.push(Sample { input: vec![x as f32 / 28.0, y as f32 / 28.0, value], output: vec![color[0] as f32 / 255.0] })
        }
    };

    samples
}

mod nn;
use std::{time::Instant, process::Command};

use nn::{NN, Sample, Samples};

use rand::{seq::SliceRandom, Rng};
use tekenen::{Platform, PlatformTrait, IntervalDecision, Event, Tekenen, colors, ui::*};

use preloaded::load_preloaded;

use image;

static mut ID: i32 = 0;

fn save_frame(nn: &mut Box<NN>, at: f32, path: &str) {
    let scale = 20;
    let size = 28 * scale;
    
    let mut tekenen = Tekenen::new(size, size);

    for x in 0..size {
        for y in 0..size {
            let mut c = nn.get(&vec![x as f32 / size as f32, y as f32 / size as f32, at])[0];

            if c < 0.0 { c = 0.0 }
            if c > 1.0 { c = 1.0 }

            let c = (c * 255.0) as u8;

            tekenen.set_pixel(x as i32, y as i32, [c, c, c, 255]);
        }
    }

    let buffer: &[u8] = tekenen.get_pixels();   

    let path = format!("./saves/{path}");

    println!("Saving image at: {path}");
    let path = std::path::Path::new(&path);

    image::save_buffer(&path, buffer, size as u32, size as u32, image::ColorType::Rgba8).unwrap();
}

fn save(nn: &mut Box<NN>, at: f32, num: f32) {
    let mut rng = rand::thread_rng();

    let att = format!("{num}").replace(".", "_");

    let this_id = unsafe {
        ID += 1;
        ID - 1
    };

    let path = format!("nn_image_at_{att}_{}_{}.png", this_id, rng.gen_range(0..999));

    save_frame(nn, at, &path);
}

fn save_video(nn: &mut Box<NN>) {
    let this_id = unsafe {
        ID += 1;
        ID - 1
    };

    let images = 500;

    let start = -4.5;
    let end = 4.5;

    let step = (end - start) / images as f32;

    for i in 0..images {
        let at = start + i as f32 * step;

        save_frame(nn, at, &format!("vid_{this_id}_{i}.png"))
    }

    Command::new("ffmpeg")
            .args([
                "-framerate", "5", 
                "-i", &format!("'./saves/vid_{this_id}_%d.png'"), 
                &format!("./saves/vid_{this_id}.mp4")
                ])
            .output()
            .unwrap();

    // ffmpeg -framerate 25 -i './saves/vid_0_%d.png' ./vid_a.mp4
}

pub fn basic() {
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

    let arch = [3, 27, 27, 9, 1];
    let mut nn = NN::new(&arch);

    let preloaded = load_preloaded();
    let mut training_data = Vec::new();

    for i in 0..=9 {
        training_data.append(&mut tekenen_to_sample(&preloaded[i], i as f32 - 4.5))
    }

    let mut training_iterations = 0;


    println!("All initialized!, arch: {:?}, training samples: {}", arch, training_data.len());

    let mut window = Platform::new(800, 600).unwrap();
    let mut tekenen = Tekenen::new(800, 600);

    let mut rate = 0.1;
    let mut graph = vec![];

    let mut rate_slider = widgets::Slider::new(50, 250, 500);
    rate_slider.min = -5.0;
    rate_slider.max = 5.0;

    let mut batch_slider = widgets::Slider::new(50, 250, 550);
    batch_slider.min = 0.0;
    batch_slider.max = 200.0;
    batch_slider.value = 20.0;

    let mut start = Instant::now();

    let mut active = true;

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
                        ' ' => active = !active,
                        'r' => {
                            start = Instant::now();
                            nn = NN::new(&arch);
                            graph = Vec::new();
                            training_iterations = 0;
                        },
                        's' => save(&mut nn, rate_slider.value, rate_slider.value + 4.5),
                        'v' => save_video(&mut nn),
                        _ => { }
                    }
                    println!("{char}")
                },
                Event::MouseDown { x, y } => {
                    rate_slider.mouse_down(x, y);
                    batch_slider.mouse_down(x, y);
                },
                Event::MouseMove { x, y } => {
                    rate_slider.mouse_move(x, y);
                    batch_slider.mouse_move(x, y);
                },
                Event::MouseUp { x, y } => {
                    rate_slider.mouse_up(x, y);
                    batch_slider.mouse_up(x, y);
                },
                _ => { }
            }
        }

        // draw operations
        tekenen.background(colors::GRAY);

        let score = nn.score(&training_data);

        if active {
            graph.push(score);
        }

        let running = Instant::now() - start;

        tekenen.draw_text(&format!("Score: {}", score), 450, 425);
        tekenen.draw_text(&format!("Batch size: {}", batch_slider.value as usize), 450, 450);
        tekenen.draw_text(&format!("Iteration: {}", training_iterations), 450, 475);
        tekenen.draw_text(&format!("Rate: {rate}"), 450, 500);
        tekenen.draw_text(&format!("Elapsed: {}", running.as_secs()), 450, 525);
        tekenen.draw_text(&format!("Drawing: {}", rate_slider.value), 450, 550);
        tekenen.draw_text(&format!("Arch: {:?}", arch), 450, 575);


        let mut rng = rand::thread_rng();

        let batch_size = batch_slider.value as usize;
        while !Platform::get_remaining_time().is_zero() && active {
            training_data.shuffle(&mut rng);

            for i in 0..(training_data.len() / batch_size) - 1 {
                let start = i * batch_size;
                let end = start + batch_size;
    
                nn.train_samples(&training_data[start..end], rate);
            }

            training_iterations += 1;

            if training_iterations % 100 == 0 {
                println!("{}", running.as_millis())
            }
        }

        // draw original image
        for i in 0..=9 {
            tekenen.draw_image(29 * i, 0, &preloaded[i as usize]);
        }

        // draw nn image
        let scale = 5;
        let size = 28 * scale;
        for x in 0..size {
            for y in 0..size {
                let mut c = nn.get(&vec![x as f32 / size as f32, y as f32 / size as f32, rate_slider.value])[0];

                if c < 0.0 { c = 0.0 }
                if c > 1.0 { c = 1.0 }

                let c = (c * 255.0) as u8;

                tekenen.set_pixel(x as i32 + scale * 58, y as i32, [c, c, c, 255]);
            }
        }

        for layer_i in 0..nn.layers.len() {
            let layer = &nn.layers[layer_i];
            let layer_bias_b = layer.bias_b.borrow();

            for neuron_i in 0..layer.len() {
                let mut c = layer_bias_b[neuron_i];
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
                    let mut c = connection.value_w[(curr_i, prev_i)];
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

        // Draw scroller
        rate_slider.display(&mut tekenen);
        batch_slider.display(&mut tekenen);

        // draw graph
        let x1 = 450;
        let y1 = 50;
        let size = 300;
        tekenen.rect(x1, y1, size, size, colors::WHITE);

        let step = size as f32 / graph.len() as f32;
        let mut x = x1 as f32;
        let mut px = x1;
        let mut py = y1;

        for i in 0..graph.len() {
            let max = graph[0];

            let y = y1 + size - ((graph[i] / max) * size as f32) as i32;

            x += step;
            tekenen.line(px, py, x as i32, y, colors::RED);
            px = x as i32;
            py = y;
        }

        window.display_pixels(tekenen.get_pixels());

        IntervalDecision::Repeat
    }, 10)
}

/*

// Before

3245
6542
10030
13232
16343
19527
22638
26033
29137
32536

3236
6231
9426
12429
15540
18637
21636
24735
27934
31129

// No clicks
3236
6231
9426
12429
15540
18637
21636
24735
27934
31129

// Changed layer indexing
3730
7332
10938
14537
18240
22339
26854
30950
34539
38244

3643
7234
10750
14446
18139
21833
25433
29141
32746
36628

3929
7540
11140
14629
18245
21835
25539
29130
32851
36645

// Changed Matrix indexing
3336
6644
9944
13237
16538
20037
23442
26846
30251
33638

3326
6642
9941
13638
17236
20530
23936
27343
30737
34142

3633
6939
10241
13641
16948
20235
23633
26947
30350
33946

// Using ndarray
4741
9342
13951
18557
23163
27765
32339
36956
41641
46964

4653
9255
13863
18539
23162
27860
32541
37257
42038
47042

4839
9452
14053
18739
23356
28163
32862
37837
42563
47238

// matrix multiplication
3334
6434
9625
12827
15938
19227
22936
26647
29934
33238

3931
7643
11342
16551
20446
24337
28352
32239
35942
39531
 */