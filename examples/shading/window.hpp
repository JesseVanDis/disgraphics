#ifndef SIMULATOR_WINDOW_HPP
#define SIMULATOR_WINDOW_HPP

#include <span>
#include <string_view>
#include <tuple>
#include <cinttypes>

namespace sim::window
{
	bool create(size_t width, size_t height, std::string_view title, size_t scale = 1);
	void draw(std::span<const uint32_t> pixels);
	void close();

	bool is_key_pressed_w();
	bool is_key_pressed_s();
	bool is_key_pressed_a();
	bool is_key_pressed_d();
	bool is_mouse_pressed();
	std::pair<double, double> get_mouse_pos();

	bool is_close_requested();
};


#endif //SIMULATOR_WINDOW_HPP
