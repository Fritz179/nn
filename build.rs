use preloader::*;

fn main() {
    // preload("src/preloaded.rs", preload_object(vec![
    //     ("img8", preload_image("src/mnist/8.png")),
    //     ("img1", preload_image("src/mnist/1.png"))
    // ]));

    let mut img_list = Vec::new();

    for i in 0..=9 {
        img_list.push(preload_image(&format!("src/mnist/{i}.png")))
    }

    preload("src/preloaded.rs", preload_array(img_list));
}