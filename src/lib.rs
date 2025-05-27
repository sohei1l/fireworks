use wasm_bindgen::prelude::*;
use web_sys::{console, window, HtmlCanvasElement, WebGlRenderingContext as GL};

// Import the `console.log` function from the browser
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

// Define a macro to provide `println!`-style syntax for console logs
macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

#[wasm_bindgen]
pub struct App {
    gl: GL,
    last_frame_time: f64,
    frame_count: u32,
}

#[wasm_bindgen]
impl App {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<App, JsValue> {
        console_log!("Initializing Neon Particles...");

        let window = window().ok_or("Failed to get window")?;
        let document = window.document().ok_or("Failed to get document")?;
        
        let canvas = document
            .get_element_by_id("canvas")
            .ok_or("Failed to find canvas element")?
            .dyn_into::<HtmlCanvasElement>()?;

        let gl = canvas
            .get_context("webgl")?
            .ok_or("Failed to get WebGL context")?
            .dyn_into::<GL>()?;

        // Set viewport to canvas size
        let width = canvas.client_width() as i32;
        let height = canvas.client_height() as i32;
        canvas.set_width(width as u32);
        canvas.set_height(height as u32);
        gl.viewport(0, 0, width, height);

        // Clear color
        gl.clear_color(0.0, 0.0, 0.0, 1.0);
        gl.clear(GL::COLOR_BUFFER_BIT);

        console_log!("WebGL context created successfully");

        Ok(App {
            gl,
            last_frame_time: 0.0,
            frame_count: 0,
        })
    }

    pub fn render(&mut self, current_time: f64) {
        // Calculate FPS every 60 frames
        if self.frame_count % 60 == 0 {
            if self.last_frame_time > 0.0 {
                let fps = 60000.0 / (current_time - self.last_frame_time);
                console_log!("FPS: {:.1}", fps);
            }
            self.last_frame_time = current_time;
        }
        self.frame_count += 1;

        // Clear the canvas
        self.gl.clear(GL::COLOR_BUFFER_BIT);
    }
}

#[wasm_bindgen(start)]
pub fn start() {
    console_log!("WASM module loaded");
}