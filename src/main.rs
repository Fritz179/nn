mod basic;

fn main () {
    let args: Vec<String> = std::env::args().collect();

    let basic = args.iter().any(|el: &String| { ["-b", "-basic", "--basic"].contains(&el.as_str()) });

    if basic {
        basic::basic();
        return;
    }

    unimplemented!("Use -b")
}