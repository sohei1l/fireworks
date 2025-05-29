use wasm_bindgen::prelude::*;
use web_sys::{window, HtmlCanvasElement, WebGlRenderingContext as GL, WebGlProgram, WebGlShader, WebGlBuffer, WebGlFramebuffer, WebGlTexture};

#[derive(Clone, Copy)]
struct Particle {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    life: f32,
    color_index: f32,
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
    // Bloom effect
    bloom_enabled: bool,
    framebuffer: WebGlFramebuffer,
    texture: WebGlTexture,
    blur_program: WebGlProgram,
    // Trails
    trails_enabled: bool,
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

        // Get canvas size (should be set by JavaScript before creating App)
        let width = canvas.width() as i32;
        let height = canvas.height() as i32;
        console_log!("Canvas dimensions: {}x{}", width, height);
        
        if width == 0 || height == 0 {
            return Err(JsValue::from_str("Canvas has zero dimensions - ensure canvas size is set before creating App"));
        }
        
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
            attribute float colorIndex;
            uniform vec2 resolution;
            varying float v_life;
            varying float v_colorIndex;
            
            void main() {
                vec2 clipspace = (position / resolution) * 2.0 - 1.0;
                gl_Position = vec4(clipspace * vec2(1, -1), 0.0, 1.0);
                gl_PointSize = 8.0;
                v_life = life;
                v_colorIndex = colorIndex;
            }
        "#).map_err(|e| JsValue::from_str(&format!("Particle vertex shader error: {}", e)))?;

        let particle_frag_shader = compile_shader(&gl, GL::FRAGMENT_SHADER, r#"
            precision mediump float;
            varying float v_life;
            varying float v_colorIndex;
            uniform vec3 u_color0;
            uniform vec3 u_color1;
            uniform vec3 u_color2;
            uniform vec3 u_color3;
            
            void main() {
                vec2 center = vec2(0.5);
                float dist = distance(gl_PointCoord, center);
                if (dist > 0.5) discard;
                
                // Cycle through colors based on life and color index
                float paletteIndex = mod(v_colorIndex + (1.0 - v_life) * 3.0, 4.0);
                
                vec3 color;
                if (paletteIndex < 1.0) {
                    color = mix(u_color0, u_color1, paletteIndex);
                } else if (paletteIndex < 2.0) {
                    color = mix(u_color1, u_color2, paletteIndex - 1.0);
                } else if (paletteIndex < 3.0) {
                    color = mix(u_color2, u_color3, paletteIndex - 2.0);
                } else {
                    color = mix(u_color3, u_color0, paletteIndex - 3.0);
                }
                
                float alpha = v_life * (1.0 - dist * 2.0);
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
                color_index: (js_sys::Math::random() as f32) * 4.0,
            });
        }

        // Create framebuffer for bloom effect
        let framebuffer = gl.create_framebuffer().ok_or("Failed to create framebuffer")?;
        let texture = gl.create_texture().ok_or("Failed to create texture")?;
        
        gl.bind_texture(GL::TEXTURE_2D, Some(&texture));
        gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_u8_array(
            GL::TEXTURE_2D, 0, GL::RGBA as i32, width, height, 0, GL::RGBA, GL::UNSIGNED_BYTE, None
        ).map_err(|e| JsValue::from_str(&format!("Texture creation error: {:?}", e)))?;
        
        gl.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_MIN_FILTER, GL::LINEAR as i32);
        gl.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_MAG_FILTER, GL::LINEAR as i32);
        gl.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_WRAP_S, GL::CLAMP_TO_EDGE as i32);
        gl.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_WRAP_T, GL::CLAMP_TO_EDGE as i32);
        
        gl.bind_framebuffer(GL::FRAMEBUFFER, Some(&framebuffer));
        gl.framebuffer_texture_2d(GL::FRAMEBUFFER, GL::COLOR_ATTACHMENT0, GL::TEXTURE_2D, Some(&texture), 0);
        gl.bind_framebuffer(GL::FRAMEBUFFER, None);

        // Create blur shader
        let blur_vert_shader = compile_shader(&gl, GL::VERTEX_SHADER, r#"
            attribute vec2 position;
            varying vec2 v_texCoord;
            
            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
                v_texCoord = (position + 1.0) * 0.5;
            }
        "#).map_err(|e| JsValue::from_str(&format!("Blur vertex shader error: {}", e)))?;

        let blur_frag_shader = compile_shader(&gl, GL::FRAGMENT_SHADER, r#"
            precision mediump float;
            varying vec2 v_texCoord;
            uniform sampler2D u_texture;
            uniform vec2 u_resolution;
            uniform float u_intensity;
            
            void main() {
                vec2 texelSize = 1.0 / u_resolution;
                vec4 color = texture2D(u_texture, v_texCoord);
                
                // Simple box blur for glow effect
                vec4 blur = vec4(0.0);
                float weight = 0.0;
                
                for (int x = -2; x <= 2; x++) {
                    for (int y = -2; y <= 2; y++) {
                        vec2 offset = vec2(float(x), float(y)) * texelSize * u_intensity;
                        blur += texture2D(u_texture, v_texCoord + offset);
                        weight += 1.0;
                    }
                }
                
                blur /= weight;
                gl_FragColor = color + blur * 0.5; // Additive blend
            }
        "#).map_err(|e| JsValue::from_str(&format!("Blur fragment shader error: {}", e)))?;

        let blur_program = link_program(&gl, &blur_vert_shader, &blur_frag_shader)
            .map_err(|e| JsValue::from_str(&format!("Blur program linking error: {}", e)))?;

        // Clear color
        gl.clear_color(0.0, 0.0, 0.0, 1.0);
        gl.enable(GL::BLEND);
        gl.blend_func(GL::SRC_ALPHA, GL::ONE);

        console_log!("WebGL setup complete with {} particles and bloom effect", particles.len());

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
            gravity_enabled: true,
            gravity: 200.0,
            friction: 0.95,
            bloom_enabled: true,
            framebuffer,
            texture,
            blur_program,
            trails_enabled: true,
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

        if self.bloom_enabled {
            // Render to framebuffer first
            self.gl.bind_framebuffer(GL::FRAMEBUFFER, Some(&self.framebuffer));
            self.gl.viewport(0, 0, self.width as i32, self.height as i32);
        }

        // Clear the canvas (or fade for trails)
        if self.trails_enabled {
            // Don't fully clear - let previous frames fade to create trails
            self.gl.clear_color(0.0, 0.0, 0.0, 0.05);
            self.gl.enable(GL::BLEND);
            self.gl.blend_func(GL::SRC_ALPHA, GL::ONE_MINUS_SRC_ALPHA);
            
            // Draw a slightly transparent black quad to fade the background
            self.gl.use_program(Some(&self.program));
            let time_location = self.gl.get_uniform_location(&self.program, "time");
            self.gl.uniform1f(time_location.as_ref(), 0.0); // Static black
            
            let resolution_location = self.gl.get_uniform_location(&self.program, "resolution");
            self.gl.uniform2f(resolution_location.as_ref(), self.width, self.height);
            
            self.gl.bind_buffer(GL::ARRAY_BUFFER, Some(&self.vertex_buffer));
            let position_location = self.gl.get_attrib_location(&self.program, "position") as u32;
            self.gl.enable_vertex_attrib_array(position_location);
            self.gl.vertex_attrib_pointer_with_i32(position_location, 2, GL::FLOAT, false, 0, 0);
            self.gl.draw_arrays(GL::TRIANGLES, 0, 6);
            
            // Reset blend mode for particles
            self.gl.blend_func(GL::SRC_ALPHA, GL::ONE);
        } else {
            // Normal clear
            self.gl.clear(GL::COLOR_BUFFER_BIT);
        }

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

        if self.bloom_enabled {
            // Render the blurred result to screen
            self.gl.bind_framebuffer(GL::FRAMEBUFFER, None);
            self.gl.viewport(0, 0, self.width as i32, self.height as i32);
            self.gl.clear(GL::COLOR_BUFFER_BIT);

            self.gl.use_program(Some(&self.blur_program));
            
            let texture_location = self.gl.get_uniform_location(&self.blur_program, "u_texture");
            let resolution_location = self.gl.get_uniform_location(&self.blur_program, "u_resolution");
            let intensity_location = self.gl.get_uniform_location(&self.blur_program, "u_intensity");
            
            self.gl.active_texture(GL::TEXTURE0);
            self.gl.bind_texture(GL::TEXTURE_2D, Some(&self.texture));
            self.gl.uniform1i(texture_location.as_ref(), 0);
            self.gl.uniform2f(resolution_location.as_ref(), self.width, self.height);
            self.gl.uniform1f(intensity_location.as_ref(), 2.0);

            // Draw full screen quad
            self.gl.bind_buffer(GL::ARRAY_BUFFER, Some(&self.vertex_buffer));
            let position_location = self.gl.get_attrib_location(&self.blur_program, "position") as u32;
            self.gl.enable_vertex_attrib_array(position_location);
            self.gl.vertex_attrib_pointer_with_i32(position_location, 2, GL::FLOAT, false, 0, 0);
            self.gl.draw_arrays(GL::TRIANGLES, 0, 6);
        }
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
                particle.color_index = (js_sys::Math::random() as f32) * 4.0;
            }
        }
    }

    fn render_particles(&mut self) {
        // Use particle shader
        self.gl.use_program(Some(&self.particle_program));

        // Set resolution uniform
        let resolution_location = self.gl.get_uniform_location(&self.particle_program, "resolution");
        self.gl.uniform2f(resolution_location.as_ref(), self.width, self.height);

        // Set color uniforms - neon colors
        let color0_location = self.gl.get_uniform_location(&self.particle_program, "u_color0");
        if let Some(location) = color0_location {
            self.gl.uniform3f(Some(&location), 1.0, 0.0, 1.0); // Magenta
        }
        let color1_location = self.gl.get_uniform_location(&self.particle_program, "u_color1");
        if let Some(location) = color1_location {
            self.gl.uniform3f(Some(&location), 0.0, 1.0, 1.0); // Cyan
        }
        let color2_location = self.gl.get_uniform_location(&self.particle_program, "u_color2");
        if let Some(location) = color2_location {
            self.gl.uniform3f(Some(&location), 0.0, 1.0, 0.0); // Green
        }
        let color3_location = self.gl.get_uniform_location(&self.particle_program, "u_color3");
        if let Some(location) = color3_location {
            self.gl.uniform3f(Some(&location), 1.0, 1.0, 0.0); // Yellow
        }

        // Prepare particle data with color index
        let mut vertex_data = Vec::new();
        for particle in &self.particles {
            vertex_data.push(particle.x);
            vertex_data.push(particle.y);
            vertex_data.push(particle.life);
            vertex_data.push(particle.color_index);
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
        let color_index_location = self.gl.get_attrib_location(&self.particle_program, "colorIndex") as u32;

        self.gl.enable_vertex_attrib_array(position_location);
        self.gl.enable_vertex_attrib_array(life_location);
        self.gl.enable_vertex_attrib_array(color_index_location);

        let stride = 4 * 4; // 4 floats * 4 bytes
        self.gl.vertex_attrib_pointer_with_i32(position_location, 2, GL::FLOAT, false, stride, 0);
        self.gl.vertex_attrib_pointer_with_i32(life_location, 1, GL::FLOAT, false, stride, 8);
        self.gl.vertex_attrib_pointer_with_i32(color_index_location, 1, GL::FLOAT, false, stride, 12);

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
                color_index: (js_sys::Math::random() as f32) * 4.0,
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

    #[wasm_bindgen]
    pub fn set_bloom(&mut self, enabled: bool) {
        self.bloom_enabled = enabled;
        console_log!("Bloom {}", if enabled { "enabled" } else { "disabled" });
    }

    #[wasm_bindgen]
    pub fn set_trails(&mut self, enabled: bool) {
        self.trails_enabled = enabled;
        console_log!("Trails {}", if enabled { "enabled" } else { "disabled" });
    }
}

#[wasm_bindgen(start)]
pub fn start() {
    console_log!("WASM module loaded");
}