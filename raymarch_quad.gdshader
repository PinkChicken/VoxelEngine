shader_type spatial;
render_mode unshaded, depth_draw_always;

uniform sampler2D raymarch_color;
uniform sampler2D raymarch_depth;

void fragment() {
	vec2 uv = UV;
	vec3 col = texture(raymarch_color, uv).rgb;
	float depth = texture(raymarch_depth, uv).r;
	ALBEDO = col;
	DEPTH = depth;
}
