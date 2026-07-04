extends CanvasLayer

const MATERIAL_IDS := [1, 2, 3, 4]
const MATERIAL_NAMES := ["ROCK", "SAND", "WATER", "GLASS"]

var _player: CharacterBody3D
var _material_db: MaterialDB
var _buttons: Array[Button] = []
var _selected_index: int = 1
var _paused_label: Label


func _ready() -> void:
	_player = get_node("/root/world/CharacterBody3D")
	_material_db = get_node("/root/world/MaterialDB")

	_build_ui()
	_player.material_changed.connect(_on_material_changed)
	_update_selection()


func _build_ui() -> void:
	var margin = MarginContainer.new()
	margin.set_anchors_preset(Control.PRESET_CENTER_BOTTOM)
	margin.offset_bottom = -20
	margin.offset_top = -70
	margin.offset_left = -200
	margin.offset_right = 200
	add_child(margin)

	var hbox = HBoxContainer.new()
	hbox.alignment = BoxContainer.ALIGNMENT_CENTER
	hbox.add_theme_constant_override("separation", 8)
	margin.add_child(hbox)

	for i in range(MATERIAL_IDS.size()):
		var mat_id = MATERIAL_IDS[i]
		var def = _material_db.get_def(mat_id)

		var btn = Button.new()
		btn.custom_minimum_size = Vector2(80, 50)
		btn.text = MATERIAL_NAMES[i]

		var style_normal = StyleBoxFlat.new()
		style_normal.bg_color = def.color if def else Color.GRAY
		style_normal.bg_color.a = 0.7
		style_normal.corner_radius_top_left = 4
		style_normal.corner_radius_top_right = 4
		style_normal.corner_radius_bottom_left = 4
		style_normal.corner_radius_bottom_right = 4
		style_normal.content_margin_left = 8
		style_normal.content_margin_right = 8
		style_normal.content_margin_top = 4
		style_normal.content_margin_bottom = 4

		var style_hover = style_normal.duplicate()
		style_hover.bg_color.a = 0.9

		var style_pressed = style_normal.duplicate()
		style_pressed.bg_color.a = 1.0

		btn.add_theme_stylebox_override("normal", style_normal)
		btn.add_theme_stylebox_override("hover", style_hover)
		btn.add_theme_stylebox_override("pressed", style_pressed)
		btn.add_theme_font_size_override("font_size", 14)

		var label_color = Color.WHITE if def and def.color.get_luminance() < 0.5 else Color.BLACK
		btn.add_theme_color_override("font_color", label_color)
		btn.add_theme_color_override("font_hover_color", label_color)
		btn.add_theme_color_override("font_pressed_color", label_color)

		var index = i
		btn.pressed.connect(func(): _select_by_index(index))
		hbox.add_child(btn)
		_buttons.append(btn)

	_paused_label = Label.new()
	_paused_label.text = "SIM PAUSED  [P]"
	_paused_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	_paused_label.add_theme_font_size_override("font_size", 18)
	_paused_label.add_theme_color_override("font_color", Color(1.0, 0.85, 0.1, 1.0))
	_paused_label.set_anchors_preset(Control.PRESET_CENTER_TOP)
	_paused_label.offset_top = 20
	_paused_label.offset_bottom = 50
	_paused_label.offset_left = -150
	_paused_label.offset_right = 150
	_paused_label.visible = false
	add_child(_paused_label)

	var hint = Label.new()
	hint.text = "1-4 or scroll to select"
	hint.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	hint.add_theme_font_size_override("font_size", 12)
	hint.add_theme_color_override("font_color", Color(1, 1, 1, 0.5))
	hint.offset_top = -18
	hint.offset_bottom = -2
	hint.offset_left = -200
	hint.offset_right = 200
	add_child(hint)


func _select_by_index(index: int) -> void:
	_player.selected_material = MATERIAL_IDS[index]
	_player.material_changed.emit(MATERIAL_IDS[index])
	_selected_index = index
	_update_selection()


func set_paused(is_paused: bool) -> void:
	_paused_label.visible = is_paused


func _on_material_changed(material_id: int) -> void:
	for i in range(MATERIAL_IDS.size()):
		if MATERIAL_IDS[i] == material_id:
			_selected_index = i
			break
	_update_selection()


func _update_selection() -> void:
	for i in range(_buttons.size()):
		var btn = _buttons[i]
		var style = btn.get_theme_stylebox("normal") as StyleBoxFlat
		if style:
			if i == _selected_index:
				style.border_width_bottom = 3
				style.border_width_top = 3
				style.border_width_left = 3
				style.border_width_right = 3
				style.border_color = Color.WHITE
			else:
				style.border_width_bottom = 0
				style.border_width_top = 0
				style.border_width_left = 0
				style.border_width_right = 0
		btn.queue_redraw()
