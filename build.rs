use preloader::*;

use std::path::Path;
use std::fs;

fn read_imgs(path: &str) -> Result<Preloaded, std::io::Error> {
    let mut entries = Vec::new();

    for i in 0..=9 {

        let mut current = Vec::new();
        for file in fs::read_dir(Path::new(&format!("{path}/{i}")))? {
            let file = file?;

            println!("{:?}", file.path());

            current.push(preload_image(file.path().to_str().unwrap()));

            if current.len() > 15 {
                break
            }
        }

        entries.push(preload_array(current))
    }

    Ok(preload_array(entries))
}

fn main() {
    // preload("src/preloaded.rs", preload_object(vec![
    //     ("img8", preload_image("src/mnist/8.png")),
    //     ("img1", preload_image("src/mnist/1.png"))
    // ]));

    let testing = read_imgs("./src/mnist/testing").unwrap();
    let training = read_imgs("./src/mnist/training").unwrap();

    preload("src/preloaded.rs", preload_object(vec![
        ("testing", testing),
        ("training", training),
    ]));
}