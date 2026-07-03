extends CharacterBody3D

signal add_sphere(center: Vector3i, radius: int, material: int)
signal remove_sphere(center: Vector3i, radius: int)

var selected_material: int = 2

signal material_changed(material_id: int)
