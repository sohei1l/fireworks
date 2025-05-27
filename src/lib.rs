use wasm_bindgen::prelude::*;
use web_sys::{console, window, HtmlCanvasElement, WebGlRenderingContext as GL, WebGlProgram, WebGlShader, WebGlBuffer};

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

fn compile_shader(gl: &GL, shader_type: u32, source: &str) -> Result<WebGlShader, String> {
    let shader = gl.create_shader(shader_type).ok_or("Failed to create shader")?;
    gl.shader_source(&shader, source);
    gl.compile_shader(&shader);

    if gl.get_shader_parameter(&shader, GL::COMPILE_STATUS).as_bool().unwrap_or(false) {
        Ok(shader)
    } else {
        Err(gl.get_shader_info_log(&shader).unwrap_or_else(|| "Unknown error creating shader".into()))
    }
}

fn link_program(gl: &GL, vert_shader: &WebGlShader, frag_shader: &WebGlShader) -> Result<WebGlProgram, String> {
    let program = gl.create_program().ok_or("Failed to create program")?;
    gl.attach_shader(&program, vert_shader);
    gl.attach_shader(&program, frag_shader);
    gl.link_program(&program);

    if gl.get_program_parameter(&program, GL::LINK_STATUS).as_bool().unwrap_or(false) {
        Ok(program)
    } else {
        Err(gl.get_program_info_log(&program).unwrap_or_else(|| "Unknown error linking program".into()))
    }
}

#[wasm_bindgen]
pub struct App {
    gl: GL,
    last_frame_time: f64,
    frame_count: u32,
    program: WebGlProgram,
    vertex_buffer: WebGlBuffer,
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

        // Create shaders
        let vert_shader = compile_shader(&gl, GL::VERTEX_SHADER, r#"
            attribute vec2 position;
            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
            }
        "#).map_err(|e| JsValue::from_str(&format!("Vertex shader error: {}", e)))?;

        let frag_shader = compile_shader(&gl, GL::FRAGMENT_SHADER, r#"
            precision mediump float;
            uniform float time;
            uniform vec2 resolution;
            
            void main() {
                vec2 uv = gl_FragCoord.xy / resolution.xy;
                vec3 color = vec3(
                    0.5 + 0.5 * sin(time * 0.001 + uv.x * 3.0),
                    0.5 + 0.5 * sin(time * 0.002 + uv.y * 3.0),
                    0.5 + 0.5 * sin(time * 0.003 + (uv.x + uv.y) * 3.0)
                );
                gl_FragColor = vec4(color * 0.3, 1.0);
            }
        "#).map_err(|e| JsValue::from_str(&format!("Fragment shader error: {}", e)))?;

        let program = link_program(&gl, &vert_shader, &frag_shader)
            .map_err(|e| JsValue::from_str(&format!("Program linking error: {}", e)))?;

        // Create fullscreen quad
        let vertices: Vec<f32> = vec![
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
            -1.0,  1.0,
             1.0, -1.0,
             1.0,  1.0,
        ];

        let vertex_buffer = gl.create_buffer().ok_or("Failed to create buffer")?;
        gl.bind_buffer(GL::ARRAY_BUFFER, Some(&vertex_buffer));

        unsafe {
            let vert_array = js_sys::Float32Array::view(&vertices);
            gl.buffer_data_with_array_buffer_view(GL::ARRAY_BUFFER, &vert_array, GL::STATIC_DRAW);
        }

        // Clear color
        gl.clear_color(0.0, 0.0, 0.0, 1.0);

        console_log!("WebGL setup complete with gradient shader");

        Ok(App {
            gl,
            last_frame_time: 0.0,
            frame_count: 0,
            program,
            vertex_buffer,
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

        // Use our shader program
        self.gl.use_program(Some(&self.program));

        // Set uniforms
        let time_location = self.gl.get_uniform_location(&self.program, "time");
        self.gl.uniform1f(time_location.as_ref(), current_time as f32);

        let resolution_location = self.gl.get_uniform_location(&self.program, "resolution");
        self.gl.uniform2f(resolution_location.as_ref(), 
            self.gl.drawing_buffer_width() as f32, 
            self.gl.drawing_buffer_height() as f32);

        // Bind vertex buffer and set up position attribute
        self.gl.bind_buffer(GL::ARRAY_BUFFER, Some(&self.vertex_buffer));
        let position_location = self.gl.get_attrib_location(&self.program, "position") as u32;
        self.gl.enable_vertex_attrib_array(position_location);
        self.gl.vertex_attrib_pointer_with_i32(position_location, 2, GL::FLOAT, false, 0, 0);

        // Draw the quad
        self.gl.draw_arrays(GL::TRIANGLES, 0, 6);
    }
}

#[wasm_bindgen(start)]
pub fn start() {
    console_log!("WASM module loaded");
}