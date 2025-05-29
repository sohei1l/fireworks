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
    is_magic_brush: bool,
    size_multiplier: f32,
    fade_speed: f32,
    is_trail_particle: bool,
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
    // Northern lights
    northern_lights_program: WebGlProgram,
}

#[wasm_bindgen]
impl App {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<App, JsValue> {

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

                // Create a dark night sky with subtle gradient
                float skyGradient = smoothstep(0.0, 1.0, uv.y);
                vec3 darkBlue = vec3(0.02, 0.02, 0.08); // Very dark blue at bottom
                vec3 midnightBlue = vec3(0.01, 0.01, 0.05); // Even darker at top

                vec3 skyColor = mix(darkBlue, midnightBlue, skyGradient);

                // Remove the background shader stars - they're causing the purple dots issue

                gl_FragColor = vec4(skyColor, 1.0);
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
            attribute float isMagicBrush;
            attribute float sizeMultiplier;
            uniform vec2 resolution;
            varying float v_life;
            varying float v_colorIndex;
            varying float v_isMagicBrush;

            void main() {
                vec2 clipspace = (position / resolution) * 2.0 - 1.0;
                gl_Position = vec4(clipspace * vec2(1, -1), 0.0, 1.0);

                // Base size with variation (reduced by 25%)
                float baseSize;
                if (isMagicBrush > 0.5) {
                    baseSize = 13.5 + life * 9.0; // Bigger magic brush particles
                } else {
                    baseSize = 6.0 + life * 3.0; // Smaller regular particles
                }

                gl_PointSize = baseSize * sizeMultiplier;

                v_life = life;
                v_colorIndex = colorIndex;
                v_isMagicBrush = isMagicBrush;
            }
        "#).map_err(|e| JsValue::from_str(&format!("Particle vertex shader error: {}", e)))?;

        let particle_frag_shader = compile_shader(&gl, GL::FRAGMENT_SHADER, r#"
            precision mediump float;
            varying float v_life;
            varying float v_colorIndex;
            varying float v_isMagicBrush;
            uniform vec3 u_color0;
            uniform vec3 u_color1;
            uniform vec3 u_color2;
            uniform vec3 u_color3;
            uniform float u_time;

            void main() {
                vec2 center = vec2(0.5);
                float dist = distance(gl_PointCoord, center);
                if (dist > 0.5) discard;

                if (v_isMagicBrush > 0.5) {
                    // Light painting photography effect - warm yellowish light colors
                    vec3 warmYellow = vec3(1.0, 0.9, 0.3);      // Warm yellow
                    vec3 brightYellow = vec3(1.0, 1.0, 0.4);    // Bright yellow
                    vec3 orangeYellow = vec3(1.0, 0.7, 0.2);    // Orange yellow
                    vec3 creamWhite = vec3(1.0, 0.95, 0.8);     // Cream white
                    vec3 goldYellow = vec3(1.0, 0.8, 0.1);      // Gold yellow
                    vec3 paleYellow = vec3(1.0, 0.98, 0.7);     // Pale yellow

                    // Yellowish light painting color selection
                    vec3 color;
                    float lightSelector = mod(v_colorIndex - 8.0, 6.0); // Adjust for new color range

                    if (lightSelector < 1.0) {
                        color = mix(warmYellow, brightYellow, lightSelector);
                    } else if (lightSelector < 2.0) {
                        color = mix(brightYellow, orangeYellow, lightSelector - 1.0);
                    } else if (lightSelector < 3.0) {
                        color = mix(orangeYellow, creamWhite, lightSelector - 2.0);
                    } else if (lightSelector < 4.0) {
                        color = mix(creamWhite, goldYellow, lightSelector - 3.0);
                    } else if (lightSelector < 5.0) {
                        color = mix(goldYellow, paleYellow, lightSelector - 4.0);
                    } else {
                        color = mix(paleYellow, warmYellow, lightSelector - 5.0);
                    }

                    // Intense glow effect like long exposure photography
                    float glow = 1.0 - dist * 0.8; // Softer falloff for glow
                    glow = pow(glow, 0.5); // Gamma correction for realistic light

                    // Photography-style light painting with bloom
                    float coreIntensity = 1.0 - dist * 2.0; // Bright core
                    float bloomIntensity = (1.0 - dist * 0.3) * 0.6; // Wide bloom

                    float totalIntensity = max(coreIntensity, bloomIntensity) * v_life;

                    gl_FragColor = vec4(color * 1.5, totalIntensity); // Brighter for light painting effect
                } else {
                    // Background particles are stars that twinkle, or trail particles
                    float time = u_time * 0.001;

                    // Only check for trail particles if they have the right characteristics
                    bool isTrail = v_colorIndex < 1.5 && v_life < 0.8 && v_life > 0.1; // More specific trail detection

                    if (isTrail) {
                        // Trail particles - bright white/yellow, fade quickly
                        vec3 trailColor = mix(vec3(1.0, 1.0, 0.8), vec3(1.0, 0.8, 0.4), v_life);
                        float trailAlpha = (1.0 - dist * 2.0) * v_life * v_life; // Quadratic fade
                        trailAlpha = max(0.0, trailAlpha);
                        gl_FragColor = vec4(trailColor, trailAlpha);
                    } else {
                        // Regular stars - realistic twinkling
                        // Much slower, more subtle twinkling based on position
                        float twinkleSpeed = 0.5 + v_colorIndex * 0.3; // Different stars twinkle at different rates
                        float twinkle = sin(time * twinkleSpeed + v_colorIndex * 6.28) * 0.3 + 0.7; // Subtler variation

                        // More varied star colors - from blue-white to yellow-white to red
                        vec3 starColor;
                        float colorTemp = v_colorIndex * 0.3; // Color temperature variation
                        if (colorTemp < 0.3) {
                            starColor = mix(vec3(0.9, 0.9, 1.0), vec3(1.0, 1.0, 1.0), colorTemp * 3.33); // Blue-white
                        } else if (colorTemp < 0.6) {
                            starColor = mix(vec3(1.0, 1.0, 1.0), vec3(1.0, 0.95, 0.8), (colorTemp - 0.3) * 3.33); // White to warm
                        } else {
                            starColor = mix(vec3(1.0, 0.95, 0.8), vec3(1.0, 0.8, 0.6), (colorTemp - 0.6) * 2.5); // Warm to orange
                        }

                        // Stars appear as tiny points with subtle twinkling
                        float starAlpha = (1.0 - dist * 4.0) * v_life * twinkle;
                        starAlpha = max(0.0, starAlpha);

                        gl_FragColor = vec4(starColor, starAlpha);
                    }
                }
            }
        "#).map_err(|e| JsValue::from_str(&format!("Particle fragment shader error: {}", e)))?;

        let particle_program = link_program(&gl, &particle_vert_shader, &particle_frag_shader)
            .map_err(|e| JsValue::from_str(&format!("Particle program linking error: {}", e)))?;

        // Create particle buffer for dynamic data
        let particle_buffer = gl.create_buffer().ok_or("Failed to create particle buffer")?;

        // Initialize particles with realistic star distribution
        let mut particles = Vec::new();

        // Create realistic star field with varied density
        for i in 0..300 {
            // Create clusters and sparse areas like real night sky
            let cluster_factor = (js_sys::Math::random() as f32).powf(1.8); // Less sparse, more stars

            if cluster_factor > 0.25 { // Lower threshold for more stars
                let is_shooting = i < 3; // Only 3 shooting stars for realism

                // For shooting stars, position them at different starting points
                let (x, y) = if is_shooting {
                    // Start shooting stars from left side at different heights to avoid overlap
                    let start_x = -50.0; // Start off-screen left
                    let start_y = 50.0 + (i as f32) * (height as f32 * 0.3); // Space them vertically
                    (start_x, start_y)
                } else {
                    // Random positioning with some clustering for regular stars
                    let x = (js_sys::Math::random() as f32) * width as f32;
                    let y = (js_sys::Math::random() as f32) * height as f32;
                    (x, y)
                };

                particles.push(Particle {
                    x,
                    y,
                    vx: if is_shooting {
                        120.0 + (js_sys::Math::random() as f32) * 40.0 // Consistent fast rightward speed
                    } else { 0.0 },
                    vy: if is_shooting {
                        -20.0 + (js_sys::Math::random() as f32) * 10.0 // Slight downward angle variation
                    } else { 0.0 },
                    life: if is_shooting {
                        0.7 + (js_sys::Math::random() as f32) * 0.3 // Shooting stars start with varied life
                    } else {
                        0.8 + (js_sys::Math::random() as f32) * 0.2 // Background stars have varied brightness
                    },
                    color_index: (js_sys::Math::random() as f32) * 4.0,
                    is_magic_brush: false,
                    size_multiplier: if is_shooting {
                        1.2 + (js_sys::Math::random() as f32) * 1.5 // Very varied shooting star sizes
                    } else {
                        0.2 + (js_sys::Math::random() as f32) * 1.0 // Much more varied star sizes
                    },
                    fade_speed: if is_shooting {
                        0.2 + (js_sys::Math::random() as f32) * 0.4 // Varied shooting star fade
                    } else {
                        0.05 + (js_sys::Math::random() as f32) * 0.15 // Very slow, varied star twinkle
                    },
                    is_trail_particle: false,
                });
            }
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

        // Create northern lights background shader (replacing artifact shader)
        let northern_lights_vert_shader = compile_shader(&gl, GL::VERTEX_SHADER, r#"
            attribute vec2 position;
            varying vec2 v_texCoord;

            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
                v_texCoord = (position + 1.0) * 0.5;
            }
        "#).map_err(|e| JsValue::from_str(&format!("Northern lights vertex shader error: {}", e)))?;

        let northern_lights_frag_shader = compile_shader(&gl, GL::FRAGMENT_SHADER, r#"
            precision mediump float;
            varying vec2 v_texCoord;
            uniform float u_time;
            uniform vec2 u_resolution;

            void main() {
                vec2 uv = v_texCoord;
                float time = u_time * 0.00008; // Much slower, more realistic movement

                // More subtle aurora bands with gentle movement
                float curtain1 = sin(uv.x * 1.8 + time * 0.8) * cos(uv.x * 1.2 + time * 0.6) * 0.3;
                float curtain2 = sin(uv.x * 2.2 - time * 0.5) * cos(uv.x * 1.6 - time * 0.4) * 0.25;

                // Add gentle flowing organic movement
                float organic1 = sin(uv.x * 3.0 + time * 0.7) * sin(uv.y * 2.0 + time * 0.3) * 0.15;
                float organic2 = cos(uv.x * 2.5 - time * 0.4) * cos(uv.y * 1.8 - time * 0.2) * 0.12;

                // Aurora spans across more of the screen with gentle movement
                float screenLimit = smoothstep(0.9, 0.1, uv.x + sin(time * 0.2) * 0.05); // Subtle moving boundary across more screen

                // Two aurora bands with more vertical separation
                float aurora1_height = 0.20 + curtain1 * 0.12 + organic1;
                float aurora2_height = 0.50 + curtain2 * 0.10 + organic2;

                // Dynamic thickness that changes more dramatically
                float thickness1 = 0.05 + sin(time * 1.2 + uv.x * 3.0) * 0.04;
                float thickness2 = 0.07 + cos(time * 1.0 - uv.x * 2.5) * 0.05;

                // Calculate fade for each aurora band
                float fade1 = smoothstep(aurora1_height - thickness1, aurora1_height - thickness1 * 0.3, uv.y) *
                             smoothstep(aurora1_height + thickness1, aurora1_height + thickness1 * 0.3, uv.y);

                float fade2 = smoothstep(aurora2_height - thickness2, aurora2_height - thickness2 * 0.3, uv.y) *
                             smoothstep(aurora2_height + thickness2, aurora2_height + thickness2 * 0.3, uv.y);

                // Apply screen limitation
                fade1 *= screenLimit;
                fade2 *= screenLimit;

                // More dynamic green to blue gradient colors
                vec3 brightGreen = vec3(0.0, 1.0, 0.4);   // Bright green
                vec3 deepGreen = vec3(0.0, 0.8, 0.2);    // Deep green
                vec3 blueGreen = vec3(0.0, 0.9, 0.7);    // Blue-green
                vec3 skyBlue = vec3(0.2, 0.7, 1.0);      // Sky blue

                // Faster color transitions
                vec3 color1 = mix(brightGreen, blueGreen, sin(time * 1.5 + uv.x * 2.0) * 0.5 + 0.5);
                vec3 color2 = mix(deepGreen, skyBlue, sin(time * 1.8 - uv.x * 2.5) * 0.5 + 0.5);

                // Combine aurora bands
                vec3 finalColor = color1 * fade1 + color2 * fade2;

                // More dynamic vertical streaks
                float streaks = sin(uv.x * 25.0 + time * 4.0) * sin(uv.y * 12.0 + time * 3.0) * 0.08 + 0.92;

                // More pronounced breathing effect
                float intensity = 0.5 + sin(time * 0.6) * 0.25;

                float totalAlpha = (fade1 + fade2) * intensity * streaks;
                gl_FragColor = vec4(finalColor, totalAlpha);
            }
        "#).map_err(|e| JsValue::from_str(&format!("Northern lights fragment shader error: {}", e)))?;

        let northern_lights_program = link_program(&gl, &northern_lights_vert_shader, &northern_lights_frag_shader)
            .map_err(|e| JsValue::from_str(&format!("Northern lights program linking error: {}", e)))?;


        // Clear color
        gl.clear_color(0.0, 0.0, 0.0, 1.0);
        gl.enable(GL::BLEND);
        gl.blend_func(GL::SRC_ALPHA, GL::ONE);


        Ok(App {
            gl,
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
            friction: 0.95,
            bloom_enabled: true,
            framebuffer,
            texture,
            blur_program,
            trails_enabled: true,
            northern_lights_program,
        })
    }

    pub fn render(&mut self, current_time: f64) {
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

        // Render northern lights background
        self.render_northern_lights(current_time);

        // Update and render particles
        self.update_particles(current_time);
        self.render_particles(current_time);

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
        let mut new_trail_particles = Vec::new();

        for particle in &mut self.particles {
            if particle.is_trail_particle {
                // Trail particles fade quickly and don't move much
                particle.life -= 0.02;
                particle.vx *= 0.95;
                particle.vy *= 0.95;
                particle.x += particle.vx * dt;
                particle.y += particle.vy * dt;
            } else if particle.is_magic_brush {
                // Light painting particles - smooth, flowing movement
                if self.gravity_enabled {
                    particle.vy += self.gravity * dt * 0.05; // Almost no gravity for light painting
                }

                particle.vx *= 0.995; // Very minimal air resistance for smooth trails
                particle.vy *= 0.995;

                particle.x += particle.vx * dt;
                particle.y += particle.vy * dt;

                // Light painting fade (20 seconds): 1.0 / (20 * 60) = 0.00083
                particle.life -= 0.00083 * particle.fade_speed;
            } else if particle.vx.abs() > 50.0 && particle.vy.abs() > 5.0 {
                // Shooting star movement
                particle.x += particle.vx * dt;
                particle.y += particle.vy * dt;
                
                // Fade shooting stars as they move
                particle.life -= 0.008 * particle.fade_speed;
                
                // Create trail particles for shooting star effect
                if particle.life > 0.1 && (js_sys::Math::random() as f32) < 0.6 {
                    let trail_particle = Particle {
                        x: particle.x + (js_sys::Math::random() as f32 - 0.5) * 3.0,
                        y: particle.y + (js_sys::Math::random() as f32 - 0.5) * 3.0,
                        vx: particle.vx * 0.3 + (js_sys::Math::random() as f32 - 0.5) * 10.0,
                        vy: particle.vy * 0.3 + (js_sys::Math::random() as f32 - 0.5) * 10.0,
                        life: 0.8,
                        color_index: 1.0, // Bright white for trail
                        is_magic_brush: false,
                        size_multiplier: 0.3 + (js_sys::Math::random() as f32) * 0.4,
                        fade_speed: 2.0 + (js_sys::Math::random() as f32) * 2.0,
                        is_trail_particle: true,
                    };
                    new_trail_particles.push(trail_particle);
                }
                
                // Remove shooting stars when they go off screen
                if particle.x > self.width + 100.0 || particle.y > self.height + 100.0 || particle.y < -100.0 {
                    particle.life = 0.0; // Mark for removal
                }
            } else {
                // Regular stars - completely static, just twinkling
                // No movement at all - real stars don't move noticeably

                // Stars very slowly vary in brightness (much more subtle)
                particle.life -= 0.0002 * particle.fade_speed;

                // Prevent stars from fading completely - they should stay visible
                if particle.life < 0.6 {
                    particle.life = 0.6 + (js_sys::Math::random() as f32) * 0.4;
                }
            }

            // Particles with life <= 0.0 will be removed by the retain filter below
        }

        // Add new trail particles
        self.particles.extend(new_trail_particles);

        // Remove particles that have completely faded out (life <= 0)
        self.particles.retain(|particle| particle.life > 0.0);
    }

    fn render_particles(&mut self, current_time: f64) {
        // Use particle shader
        self.gl.use_program(Some(&self.particle_program));

        // Set resolution uniform
        let resolution_location = self.gl.get_uniform_location(&self.particle_program, "resolution");
        self.gl.uniform2f(resolution_location.as_ref(), self.width, self.height);

        // Set time uniform for twinkling stars
        let time_location = self.gl.get_uniform_location(&self.particle_program, "u_time");
        if let Some(location) = time_location {
            self.gl.uniform1f(Some(&location), current_time as f32);
        }

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

        // Prepare particle data with all attributes
        let mut vertex_data = Vec::new();
        for particle in &self.particles {
            vertex_data.push(particle.x);
            vertex_data.push(particle.y);
            vertex_data.push(particle.life);
            vertex_data.push(particle.color_index);
            vertex_data.push(if particle.is_magic_brush { 1.0 } else { 0.0 });
            vertex_data.push(particle.size_multiplier);
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
        let magic_brush_location = self.gl.get_attrib_location(&self.particle_program, "isMagicBrush") as u32;
        let size_location = self.gl.get_attrib_location(&self.particle_program, "sizeMultiplier") as u32;

        self.gl.enable_vertex_attrib_array(position_location);
        self.gl.enable_vertex_attrib_array(life_location);
        self.gl.enable_vertex_attrib_array(color_index_location);
        self.gl.enable_vertex_attrib_array(magic_brush_location);
        self.gl.enable_vertex_attrib_array(size_location);

        let stride = 6 * 4; // 6 floats * 4 bytes
        self.gl.vertex_attrib_pointer_with_i32(position_location, 2, GL::FLOAT, false, stride, 0);
        self.gl.vertex_attrib_pointer_with_i32(life_location, 1, GL::FLOAT, false, stride, 8);
        self.gl.vertex_attrib_pointer_with_i32(color_index_location, 1, GL::FLOAT, false, stride, 12);
        self.gl.vertex_attrib_pointer_with_i32(magic_brush_location, 1, GL::FLOAT, false, stride, 16);
        self.gl.vertex_attrib_pointer_with_i32(size_location, 1, GL::FLOAT, false, stride, 20);

        // Draw particles as points
        self.gl.draw_arrays(GL::POINTS, 0, self.particles.len() as i32);
    }

    fn render_northern_lights(&mut self, current_time: f64) {
        // Use northern lights shader
        self.gl.use_program(Some(&self.northern_lights_program));

        // Set uniforms
        let time_location = self.gl.get_uniform_location(&self.northern_lights_program, "u_time");
        if let Some(location) = time_location {
            self.gl.uniform1f(Some(&location), current_time as f32);
        }

        let resolution_location = self.gl.get_uniform_location(&self.northern_lights_program, "u_resolution");
        if let Some(location) = resolution_location {
            self.gl.uniform2f(Some(&location), self.width, self.height);
        }

        // Use the same vertex buffer as the background
        self.gl.bind_buffer(GL::ARRAY_BUFFER, Some(&self.vertex_buffer));
        let position_location = self.gl.get_attrib_location(&self.northern_lights_program, "position") as u32;
        self.gl.enable_vertex_attrib_array(position_location);
        self.gl.vertex_attrib_pointer_with_i32(position_location, 2, GL::FLOAT, false, 0, 0);

        // Set blend mode for northern lights overlay
        self.gl.blend_func(GL::SRC_ALPHA, GL::ONE);

        // Draw the northern lights
        self.gl.draw_arrays(GL::TRIANGLES, 0, 6);
    }

    #[wasm_bindgen]
    pub fn spawn_particles(&mut self, x: f32, y: f32, current_time: f64) {

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
                is_magic_brush: false,
                is_trail_particle: false,
                size_multiplier: 0.5 + (js_sys::Math::random() as f32) * 1.5,
                fade_speed: 0.8 + (js_sys::Math::random() as f32) * 1.0, // Random fade speed (0.8-1.8x)
            };

            // Always add new particles - let them fade naturally based on age
            self.particles.push(new_particle);
        }
    }

    #[wasm_bindgen]
    pub fn set_gravity(&mut self, enabled: bool) {
        self.gravity_enabled = enabled;
    }

    #[wasm_bindgen]
    pub fn set_friction(&mut self, friction: f32) {
        self.friction = friction.max(0.9).min(1.0); // Clamp between 0.9 and 1.0
    }

    #[wasm_bindgen]
    pub fn set_bloom(&mut self, enabled: bool) {
        self.bloom_enabled = enabled;
    }

    #[wasm_bindgen]
    pub fn set_trails(&mut self, enabled: bool) {
        self.trails_enabled = enabled;
    }

    #[wasm_bindgen]
    pub fn spawn_magic_brush_particles(&mut self, x: f32, y: f32, current_time: f64) {
        // Spawn more particles for better continuous drawing
        for _ in 0..6 {
            let angle = js_sys::Math::random() as f32 * 2.0 * std::f32::consts::PI;
            let speed = (js_sys::Math::random() as f32) * 15.0 + 5.0; // Slow speed for light painting

            // Small spread for continuous light painting effect
            let offset_x = (js_sys::Math::random() as f32 - 0.5) * 8.0;
            let offset_y = (js_sys::Math::random() as f32 - 0.5) * 8.0;

            let new_particle = Particle {
                x: x + offset_x,
                y: y + offset_y,
                vx: angle.cos() * speed,
                vy: angle.sin() * speed,
                life: 1.0,
                color_index: 8.0 + (js_sys::Math::random() as f32) * 2.0, // Yellowish light painting color range
                is_magic_brush: true,
                is_trail_particle: false,
                size_multiplier: 0.8 + (js_sys::Math::random() as f32) * 0.4,
                fade_speed: 0.5 + (js_sys::Math::random() as f32) * 1.0, // Random fade speed for natural variation (0.5-1.5x)
            };

            // Always add new particles - let them fade naturally based on age
            self.particles.push(new_particle);
        }
    }
}

#[wasm_bindgen(start)]
pub fn start() {
}