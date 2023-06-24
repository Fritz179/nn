use preloader::*;

fn main() {
    preload("src/preloaded.rs", preload_object(vec![
        ("img8", preload_image("src/mnist/8.png")),
        ("img1", preload_image("src/mnist/1.png"))
    ]));
}