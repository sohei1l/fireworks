use wasm_bindgen::prelude::*;
use web_sys::{console, window, HtmlCanvasElement, WebGlRenderingContext as GL, WebGlProgram, WebGlShader, WebGlBuffer, MouseEvent};

#[derive(Clone, Copy)]
struct Particle {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    life: f32,
    max_life: f32,
}

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
    particle_program: WebGlProgram,
    particle_buffer: WebGlBuffer,
    particles: Vec<Particle>,
    width: f32,
    height: f32,
    gravity_enabled: bool,
    gravity: f32,
    friction: f32,
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

        // Create particle shader
        let particle_vert_shader = compile_shader(&gl, GL::VERTEX_SHADER, r#"
            attribute vec2 position;
            attribute float life;
            uniform vec2 resolution;
            varying float v_life;
            
            void main() {
                vec2 clipspace = (position / resolution) * 2.0 - 1.0;
                gl_Position = vec4(clipspace * vec2(1, -1), 0.0, 1.0);
                gl_PointSize = 8.0;
                v_life = life;
            }
        "#).map_err(|e| JsValue::from_str(&format!("Particle vertex shader error: {}", e)))?;

        let particle_frag_shader = compile_shader(&gl, GL::FRAGMENT_SHADER, r#"
            precision mediump float;
            varying float v_life;
            
            void main() {
                vec2 center = vec2(0.5);
                float dist = distance(gl_PointCoord, center);
                if (dist > 0.5) discard;
                
                float alpha = v_life * (1.0 - dist * 2.0);
                vec3 color = vec3(0.0, 1.0, 1.0); // Cyan glow
                gl_FragColor = vec4(color, alpha);
            }
        "#).map_err(|e| JsValue::from_str(&format!("Particle fragment shader error: {}", e)))?;

        let particle_program = link_program(&gl, &particle_vert_shader, &particle_frag_shader)
            .map_err(|e| JsValue::from_str(&format!("Particle program linking error: {}", e)))?;

        // Create particle buffer for dynamic data
        let particle_buffer = gl.create_buffer().ok_or("Failed to create particle buffer")?;

        // Initialize particles
        let mut particles = Vec::new();
        for _ in 0..100 {
            particles.push(Particle {
                x: (js_sys::Math::random() as f32) * width as f32,
                y: (js_sys::Math::random() as f32) * height as f32,
                vx: (js_sys::Math::random() as f32 - 0.5) * 100.0,
                vy: (js_sys::Math::random() as f32 - 0.5) * 100.0,
                life: 1.0,
                max_life: 1.0,
            });
        }

        // Clear color
        gl.clear_color(0.0, 0.0, 0.0, 1.0);
        gl.enable(GL::BLEND);
        gl.blend_func(GL::SRC_ALPHA, GL::ONE);

        console_log!("WebGL setup complete with {} particles", particles.len());

        Ok(App {
            gl,
            last_frame_time: 0.0,
            frame_count: 0,
            program,
            vertex_buffer,
            particle_program,
            particle_buffer,
            particles,
            width: width as f32,
            height: height as f32,
            gravity_enabled: false,
            gravity: 200.0,
            friction: 0.99,
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

        // Update and render particles
        self.update_particles(current_time);
        self.render_particles();
    }

    fn update_particles(&mut self, _current_time: f64) {
        let dt = 0.016; // 60fps assumption

        for particle in &mut self.particles {
            // Apply gravity
            if self.gravity_enabled {
                particle.vy += self.gravity * dt;
            }

            // Apply friction
            particle.vx *= self.friction;
            particle.vy *= self.friction;

            // Simple motion
            particle.x += particle.vx * dt;
            particle.y += particle.vy * dt;

            // Bounce off edges with some energy loss
            if particle.x < 0.0 || particle.x > self.width {
                particle.vx = -particle.vx * 0.8; // Energy loss on bounce
                particle.x = particle.x.max(0.0).min(self.width);
            }
            if particle.y < 0.0 || particle.y > self.height {
                particle.vy = -particle.vy * 0.8; // Energy loss on bounce
                particle.y = particle.y.max(0.0).min(self.height);
            }

            // Fade particles slowly
            particle.life -= 0.002;
            if particle.life <= 0.0 {
                // Respawn particle
                particle.x = (js_sys::Math::random() as f32) * self.width;
                particle.y = (js_sys::Math::random() as f32) * self.height;
                particle.vx = (js_sys::Math::random() as f32 - 0.5) * 100.0;
                particle.vy = (js_sys::Math::random() as f32 - 0.5) * 100.0;
                particle.life = 1.0;
            }
        }
    }

    fn render_particles(&mut self) {
        // Use particle shader
        self.gl.use_program(Some(&self.particle_program));

        // Set resolution uniform
        let resolution_location = self.gl.get_uniform_location(&self.particle_program, "resolution");
        self.gl.uniform2f(resolution_location.as_ref(), self.width, self.height);

        // Prepare particle data
        let mut vertex_data = Vec::new();
        for particle in &self.particles {
            vertex_data.push(particle.x);
            vertex_data.push(particle.y);
            vertex_data.push(particle.life);
        }

        // Upload particle data
        self.gl.bind_buffer(GL::ARRAY_BUFFER, Some(&self.particle_buffer));
        unsafe {
            let data_array = js_sys::Float32Array::view(&vertex_data);
            self.gl.buffer_data_with_array_buffer_view(GL::ARRAY_BUFFER, &data_array, GL::DYNAMIC_DRAW);
        }

        // Set up attributes
        let position_location = self.gl.get_attrib_location(&self.particle_program, "position") as u32;
        let life_location = self.gl.get_attrib_location(&self.particle_program, "life") as u32;

        self.gl.enable_vertex_attrib_array(position_location);
        self.gl.enable_vertex_attrib_array(life_location);

        let stride = 3 * 4; // 3 floats * 4 bytes
        self.gl.vertex_attrib_pointer_with_i32(position_location, 2, GL::FLOAT, false, stride, 0);
        self.gl.vertex_attrib_pointer_with_i32(life_location, 1, GL::FLOAT, false, stride, 8);

        // Draw particles as points
        self.gl.draw_arrays(GL::POINTS, 0, self.particles.len() as i32);
    }

    #[wasm_bindgen]
    pub fn spawn_particles(&mut self, x: f32, y: f32) {
        console_log!("Spawning particles at ({}, {})", x, y);
        
        // Spawn a burst of 10 particles at the mouse position
        for _ in 0..10 {
            let angle = js_sys::Math::random() as f32 * 2.0 * std::f32::consts::PI;
            let speed = (js_sys::Math::random() as f32) * 150.0 + 50.0;
            
            let new_particle = Particle {
                x,
                y,
                vx: angle.cos() * speed,
                vy: angle.sin() * speed,
                life: 1.0,
                max_life: 1.0,
            };

            // Replace oldest particles or extend if under limit
            if self.particles.len() < 500 {
                self.particles.push(new_particle);
            } else {
                // Replace the first particle that has low life
                if let Some(old_particle) = self.particles.iter_mut().find(|p| p.life < 0.3) {
                    *old_particle = new_particle;
                } else {
                    // Replace first particle as fallback
                    self.particles[0] = new_particle;
                }
            }
        }
    }

    #[wasm_bindgen]
    pub fn set_gravity(&mut self, enabled: bool) {
        self.gravity_enabled = enabled;
        console_log!("Gravity {}", if enabled { "enabled" } else { "disabled" });
    }

    #[wasm_bindgen]
    pub fn set_friction(&mut self, friction: f32) {
        self.friction = friction.max(0.9).min(1.0); // Clamp between 0.9 and 1.0
        console_log!("Friction set to {}", self.friction);
    }
}

#[wasm_bindgen(start)]
pub fn start() {
    console_log!("WASM module loaded");
}