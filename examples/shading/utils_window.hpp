#ifndef SIMULATOR_WINDOW_HPP
#define SIMULATOR_WINDOW_HPP

#include <span>
#include <string_view>
#include <tuple>
#include <cinttypes>
#include <memory>

namespace example::utils
{
	class input
	{
		public:
			virtual bool is_key_pressed_w() = 0;
			virtual bool is_key_pressed_s() = 0;
			virtual bool is_key_pressed_a() = 0;
			virtual bool is_key_pressed_d() = 0;
			virtual bool is_mouse_pressed() = 0;
			virtual std::pair<double, double> get_mouse_pos() = 0;
	};

	namespace detail
	{
		class window final : public input
		{
				friend bool is_ok(window& window);

				struct impl;
				std::unique_ptr<impl> m_impl;
			public:
				window(size_t width, size_t height, std::string_view title, size_t scale = 1);
				~window();
				void draw(std::span<const uint32_t> pixels);

				bool is_key_pressed_w() override;
				bool is_key_pressed_s() override;
				bool is_key_pressed_a() override;
				bool is_key_pressed_d() override;
				bool is_mouse_pressed() override;
				std::pair<double, double> get_mouse_pos() override;

				bool is_close_requested();
		};

		bool is_ok(window& window);
	}

	using window = std::unique_ptr<detail::window>;
	window create_window(size_t width, size_t height, std::string_view title, size_t scale);
};


#endif //SIMULATOR_WINDOW_HPP
