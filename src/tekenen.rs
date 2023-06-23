mod font;
use font::{FIRST_CHAR, FONT, LAST_CHAR};

pub type Pixel = [u8; 4];
pub type Pixels = Vec<u8>;

#[allow(dead_code)]
pub const WHITE: Pixel = [255, 255, 255, 255];
#[allow(dead_code)]
pub const RED: Pixel = [255, 0, 0, 255];
#[allow(dead_code)]
pub const GRAY: Pixel = [51, 51, 51, 255];
#[allow(dead_code)]
pub const BLACK: Pixel = [0, 0, 0, 255];

pub struct Tekenen {
    pixels: Pixels,
    width: usize,
    height: usize,
}

impl Tekenen {
    pub fn new(width: usize, height: usize) -> Tekenen {
        Tekenen {
            pixels: vec![0; width * height * 4],
            width,
            height,
        }
    }

    pub fn get_pixels(&self) -> &Pixels {
        &self.pixels
    }
}

// Drawing implementation
impl Tekenen {
    pub fn pixel_index(&self, x: i32, y: i32) -> Option<usize> {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
            None
        } else {
            Some((y * self.width as i32 + x) as usize)
        }
    }

    pub fn set_pixel(&mut self, x: i32, y: i32, color: Pixel) {
        if let Some(index) = self.pixel_index(x, y) {
            // self.pixels.borrow_mut()[index] = color;
            self.pixels[index * 4 + 0] = color[0];
            self.pixels[index * 4 + 1] = color[1];
            self.pixels[index * 4 + 2] = color[2];
            self.pixels[index * 4 + 3] = color[3];
        }
    }

    #[allow(dead_code)]
    pub fn rect(&mut self, x: i32, y: i32, w: i32, h: i32, color: Pixel) {
        for x in x..x + w {
            for y in y..y + h {
                self.set_pixel(x, y, color);
            }
        }
    }

    #[allow(dead_code)]
    pub fn background(&mut self, color: Pixel) {
        for x in 0..self.width {
            for y in 0..self.height {
                self.set_pixel(x as i32, y as i32, color);
            }
        }
    }

    #[allow(dead_code)]
    pub fn draw_text(&mut self, text: &str, x: i32, y: i32) -> (i32, i32) {
        const FONT_SCALE: i32 = 2;
        const FONT_SIZE: i32 = 8 * FONT_SCALE;

        let mut curr_x = 0;
        let mut curr_y = 0;

        for char in text.chars() {
            if curr_x >= 800 || char == '\n' {
                curr_x = 0;
                curr_y += FONT_SIZE;

                if char == '\n' {
                    continue;
                }
            }

            // skip whitespace
            if char == ' ' {
                curr_x += FONT_SIZE;
                continue;
            }

            // get data by finding offset in charset
            let data = FONT[char as usize - FIRST_CHAR as usize];

            for (yd, line) in data.iter().enumerate() {
                let y = y + yd as i32 * FONT_SCALE + curr_y;

                for (xd, symbol) in line.iter().enumerate() {
                    let x = x + xd as i32 * FONT_SCALE + curr_x;

                    if *symbol == ' ' {
                        continue;
                    }

                    for xf in 0..FONT_SCALE {
                        for yf in 0..FONT_SCALE {
                            self.set_pixel(x + xf, y + yf, WHITE);
                        }
                    }
                }
            }

            // increment for next character
            curr_x += FONT_SIZE;
        }

        (curr_x, curr_y)
    }

    // pub fn draw_terminal(&mut self, buffer: &str, time: u64) {
    //     let (x, y) = self.draw_text(buffer, 0, 0);

    //     const BLINKING_TIME: u64 = 500;

    //     if time % BLINKING_TIME > BLINKING_TIME / 2 {
    //         self.rect(x, y, 16, 16, WHITE)
    //     }
    // }
}
