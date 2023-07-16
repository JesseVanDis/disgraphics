#ifndef CPURAYTRACER_GT_HPP
#define CPURAYTRACER_GT_HPP

#include <cstdint>
#include <memory>

namespace gr
{
	template<typename T>
	concept floating_point = std::floating_point<std::decay_t<T>>;

	template<typename T>
	concept unsigned_integral = std::unsigned_integral<std::decay_t<T>>;

	template<typename T>
	concept unsigned_int_32 = std::same_as<uint32_t, std::decay_t<T>>;

	template<typename T>
	concept vector3 = requires(T v)
	{
		{v.x} -> floating_point;
		{v.y} -> floating_point;
		{v.z} -> floating_point;
	};

	template<typename T>
	concept triangle = requires(T v)
	{
		{v.p0} -> vector3;
		{v.p1} -> vector3;
		{v.p2} -> vector3;
	};

	template<typename T>
	concept triangle_list = requires(T v)
	{
		{v.begin()};
		{v.end()};
		{*v.begin()} -> triangle;
		{*v.end()} -> triangle;
	};

	template<typename T>
	concept triangle_color = requires(T v)
	{
		{v.color} -> unsigned_int_32;
	};

	template<typename T>
	concept camera = requires(T v)
	{
		{v.pos} 			-> vector3;
		{v.lookat} 			-> vector3;
		{v.up} 				-> vector3;
		{v.fov} 			-> floating_point;
	};

	template<typename T>
	concept render_target = requires(T v)
	{
		/// 32 byte aligned red,green,blue,depth
		{v.data()} 			-> std::same_as<uint32_t*>;

		{v.width} 			-> unsigned_integral;
		{v.height} 			-> unsigned_integral;
	};

	namespace helpers
	{
		struct vec3
		{
			float x,y,z;
		};

		struct tri
		{
			vec3 p0, p1, p2;
		};

		struct cam
		{
			vec3 pos, lookat, up;
			float fov;
		};

		struct managed_render_target
		{
			std::unique_ptr<uint32_t> 	rgbd_buffer;   // 32 byte aligned red,green,blue,depth
			size_t						width;
			size_t						height;

			uint32_t*					data() { return rgbd_buffer.get(); } // NOLINT
		};

		inline managed_render_target 	create_render_target(size_t num_pixels_width, size_t num_pixels_height);
		cam 							lookat(const vector3 auto& pos, const vector3 auto& lookat, const vector3 auto& up, const std::floating_point auto& fov = 2.0f);
	}

	namespace detail
	{
		struct impl;
	}

	struct context
	{
		~context();

		std::unique_ptr<detail::impl> impl;
	};

	context			create_context();

	void			clear(render_target auto& target);

	void 			render(context& context, 	render_target auto& target, const triangle_list auto& triangles, const camera auto& camera);

	/// for performance it is recommended to use a context if you are going to render multiple frames per second
	void 			render(						render_target auto& target, const triangle_list auto& triangles, const camera auto& camera);
}

namespace grh = gr::helpers;


// implementation
#include <span>
#include <algorithm>
#include <glm/glm.hpp>
#include <glm/ext.hpp>


namespace gr
{
	namespace detail
	{
		struct impl
		{
			std::unique_ptr<float> 	managed_depth_buffer;
			size_t					managed_depth_buffer_size = 0;
		};
	}

	context::~context() = default;


	namespace detail
	{
        struct line
        {
            const grh::vec3& p0;
            const grh::vec3& p1;
        };

#if 0

        typedef struct {char StartVertIndex, EndVertIndex;CVec2f StartPoint, EndPoint;}SLine;
        typedef struct{int YStart, Height; float X, Xit, LYPerc, LYPercIt, R, Rit;}SLineIt;

        // Game::DrawTanks - draw the tanks
void DrawSpotlightTri(float CenterX, float CenterY, float p2x, float p2y, float p3x, float p3y,
	unsigned int* BufferInOut, unsigned int BufferWidth, unsigned int BufferHeight, unsigned int BufferPitch)
{
	float p1x = CenterX, p1y = CenterY;
	float Px[] = {p1x, p2x, p3x}, Py[] = {p1y, p2y, p3y};

	union{struct{SLine l, t1, t2;}L;SLine Lines[3];};
	float Rx, Ry;

	// get highest point index, lowest point index, and remaining point index
	int Hi, Ri, Li;
	if(p2y < p1y){if(p3y < p2y){Hi = 2; if(p2y > p1y) {Li = 1; Ri = 0;} else{Li = 0; Ri = 1;}}else{Hi = 1; if(p3y > p1y) {Li = 2; Ri = 0;} else{Li = 0; Ri = 2;}}}
	else if(p3y < p1y){Hi = 2;if(p2y > p1y){Li = 1; Ri = 0;}else{Li = 0; Ri = 1;}}
	else{Hi = 0;if(p3y > p2y){Li = 2; Ri = 1;}else{Li = 1; Ri = 2;}}

	// set them
	L.l.p1x	= Px[Hi];		L.l.p1y = Py[Hi];		L.l.p2x = Px[Li];		L.l.p2y = Py[Li];
	L.t1.p1x = Px[Hi];	L.t1.p1y = Py[Hi];	L.t1.p2x = Px[Ri];	L.t1.p2y = Py[Ri];
	L.t2.p1x = Px[Ri];	L.t2.p1y = Py[Ri];	L.t2.p2x = Px[Li];	L.t2.p2y = Py[Li];
	Rx = Px[Ri]; Ry = Py[Ri];

	// useful vars
	float LlHeight = L.l.p2y - L.l.p1y, Tl1Height = L.t1.p2y - L.t1.p1y, Tl2Height = L.t2.p2y - L.t2.p1y;
	int RemainingVertDirection = ((L.l.p2x-L.l.p1x)*(Ry-L.l.p1y)-(Rx-L.l.p1x)*(L.l.p2y-L.l.p1y)) < 0.0f ? -1 : 1;
	float Cx = (p2x + p3x) * 0.5f, Cy = (p2y + p3y) * 0.5f;
	float XDist = CenterX-Cx, YDist = CenterY - Cy;
	float Dist = sqrt((XDist*XDist) + (YDist*YDist));
	float XDir = XDist/Dist, YDir = YDist/Dist;

	// create iterators
	union{struct{SLineIt Ll, Tl1, Tl2;}It;SLineIt Its[3];};

	for(int i=0; i<3; i++)
	{
		SLineIt& c = Its[i]; SLine& l = Lines[i];
		float Sx = l.p1x, Sy = l.p1y, Ex = l.p2x, Ey = l.p2y; // useful vars

		c.YStart = (int)ceil(Sy);
		c.Height = (int)ceil(Ey) - c.YStart; // set

		float Oh = 1.0f / (float)c.Height;
		c.Xit = (Ex - Sx) * Oh; // interpolation

		// calculate intencity
		c.LYPerc = (1.0f - (dot(Sx - Cx, Sy - Cy, XDir, YDir) / Dist)) * 40.0f;
		float PercEnd = (1.0f - (dot(Ex - Cx, Ey - Cy, XDir, YDir) / Dist)) * 40.0f;
		c.LYPercIt = (PercEnd - c.LYPerc) * Oh;

		float	SubPixel = ceil(Sy) - Sy;
		c.X = Sx + (c.Xit * SubPixel); // sub pixel
		c.LYPerc = c.LYPerc + (c.LYPercIt * SubPixel);
	}

	// detect left and right edge
	SLineIt* RightEdge = NULL, *LeftEdge = NULL;
	if(RemainingVertDirection > 0){LeftEdge = It.Tl1.Height <= 0 ? &It.Tl2 : &It.Tl1;RightEdge = &It.Ll;}
	else{LeftEdge = &It.Ll;RightEdge = It.Tl1.Height <= 0 ? &It.Tl2 : &It.Tl1;}

	// used for the loop
	unsigned int* Buffer = BufferInOut;
	unsigned int Width = BufferWidth;
	unsigned int Height = BufferHeight;
	unsigned int Pitch = BufferPitch;
	int LeadingEdgeStartY = It.Ll.YStart;

	// clip
	if(LeadingEdgeStartY+It.Ll.Height > SCRHEIGHT-1)
		It.Ll.Height = (SCRHEIGHT-1) - LeadingEdgeStartY;

	for(int Y=LeadingEdgeStartY; Y<LeadingEdgeStartY+It.Ll.Height; Y++)
	{
		int XLeft = (int)(LeftEdge->X), XRight = (int)(RightEdge->X);
		float Width = RightEdge->X - LeftEdge->X;
		float	SubTexel = ((float)XLeft) - LeftEdge->X;

		float LPercIt = (RightEdge->LYPerc - LeftEdge->LYPerc) / Width, LPerc = LeftEdge->LYPerc + (LPercIt * SubTexel);
		unsigned int* CurrentBufferPos = &Buffer[XLeft + (Y*Pitch)];
		if(XLeft < 0)
		{
			int Differencei = -XLeft;
			float Difference = (float)Differencei;
			CurrentBufferPos+=Differencei;
			LPerc += LPerc * Difference;
			XLeft = 0;
		}
		if(XRight > SCRWIDTH-1)
			XRight = SCRWIDTH-1;
		//(*CurrentBufferPos) = 0xffffff;

		union
		{
			unsigned int Colors[75];
			__m128i Colors4[(75/4)+1];
		};
		const int RowSize = XRight - XLeft;
		if(RowSize > 0)
			memcpy(Colors, CurrentBufferPos, sizeof(unsigned int) * RowSize);
		assert(RowSize < 75);

		int CountTo4Remaining = (RowSize % 4);
		int CountTo4 = RowSize - CountTo4Remaining;

		int ItC = 0;
		for(int X4 = 0, ItC = 0; X4 < RowSize; X4+=4, ItC++)
		{
			const __m128 SPOTLIGHTSTART4 = _mm_set_ps1(SPOTLIGHT_STARTINTENCITYF);
			const __m128 SPOTLIGHT_END_MINUS_START4 = _mm_set_ps1(ER_MINUS_SR);

			union {__m128 LPerc4; float LPerca[4];};
			LPerca[0] = LPerc; LPerca[1] = LPerca[0] + LPercIt; LPerca[2] = LPerca[1] + LPercIt; LPerca[3] = LPerca[2] + LPercIt;

			//LPerc4 = _mm_set_ps1(LPerc);
			__m128i Intens4 = _mm_cvtps_epi32(_mm_add_ps(SPOTLIGHTSTART4, _mm_mul_ps(SPOTLIGHT_END_MINUS_START4, _mm_div_ps(ONE4, LPerc4))));

			__m128i& Color4 = Colors4[ItC];

			__m128i R4 = _mm_add_epi32(_mm_and_si128(Color4, REDMASK4), _mm_slli_epi32(Intens4, 16));
			__m128i G4 = _mm_add_epi32(_mm_and_si128(Color4, GREENMASK4), _mm_slli_epi32(Intens4, 8));
			__m128i B4 = _mm_and_si128(Color4, BLUEMASK4);

			__m128i R14 = _mm_or_si128(_mm_and_si128(R4, REDMASK4), _mm_slli_epi32(_mm_mullo_epi16(BLUEMASK4, _mm_srli_epi32(R4, 24)), 16));
			__m128i G14 = _mm_or_si128(_mm_and_si128(G4, GREENMASK4), _mm_mullo_epi16(GREENMASK4, _mm_srli_epi32(G4, 16)));

			Color4 = _mm_or_si128(R14, _mm_or_si128(G14, B4));

			LPerc += (LPercIt*4.0f);
		}

		if(RowSize > 0)
			memcpy(CurrentBufferPos, Colors, sizeof(unsigned int) * RowSize);

		// interpolate
		RightEdge->X += RightEdge->Xit;	RightEdge->LYPerc += RightEdge->LYPercIt;
		LeftEdge->X += LeftEdge->Xit;		LeftEdge->LYPerc += LeftEdge->LYPercIt;

		// swap loading edges if needed
		if((Y-LeadingEdgeStartY) >= RightEdge->Height-1)
			RightEdge = &It.Tl2;
		if((Y-LeadingEdgeStartY) >= LeftEdge->Height-1)
			LeftEdge = &It.Tl2;
	}
}

#endif
        void draw_triangle(render_target auto& target, const triangle auto& source_triangle, const line& line_long, const line& line_top, const line& line_bot)
        {

        }

        inline float cross_z(float ax, float ay, float bx, float by)
        {
            return ax * by - bx * ay;
        }

        struct line_it
        {
            int y_start, height;

			float x_it, x;
			float z_it, z;

			line_it& operator ++()
			{
				x += x_it;
				z += z_it;
				return *this;
			}
        };

        line_it line_it_from_line(const vector3 auto& p0, const vector3 auto& p1)
        {
			const float y_start_ceiled 			= std::ceil(p0.y);
			const float height_ceiled 			= std::ceil(p1.y) - y_start_ceiled;
			const float one_over_height_ceiled 	= 1.0f / height_ceiled;
			const float sub_pixel 				= y_start_ceiled - p0.y;
			assert(height_ceiled != 0.0f); // this is going to be a division over 0 ! // TODO: handle this to avoid NaN

            line_it c {
                    .y_start    = static_cast<int>(y_start_ceiled),
                    .height     = static_cast<int>(height_ceiled),
					.x_it 		= (p1.x - p0.x) * one_over_height_ceiled,
					.x			= p0.x + (c.x_it * sub_pixel),
					.z_it 		= (p1.z - p0.z) * one_over_height_ceiled,
					.z			= p0.z + (c.z_it * sub_pixel)
			};
			return c;
		}

		struct draw_hline_ctx
		{
			int y, x_from, x_to;
		};

		inline void draw_hline_fill_0xff(render_target auto& target, const triangle auto& source_triangle, const draw_hline_ctx& ctx)
		{
			std::span<uint32_t> target_data{target.data(), target.width * target.height};
			auto begin = std::next(target_data.begin(), ctx.x_from + ctx.y * target.width);
			auto end = std::next(begin, ctx.x_to - ctx.x_from);
			std::generate(begin, end, 0xffffffff);
		}


        void draw_triangle(render_target auto& target, const triangle auto& source_triangle, std::array<grh::vec3, 3>& pts)
		{
            struct line {const grh::vec3& p0, &p1;};

            std::sort(pts.begin(), pts.end(), [](const grh::vec3& a, const grh::vec3& b){return a.y < b.y;});

            // take lines
            const line line_long  = {pts[0], pts[2]};
            const line line_top   = {pts[0], pts[1]};
            const line line_bot   = {pts[1], pts[2]};

            // check whether the long line is on the left or right
            float cross_z = (pts[1].x - pts[0].x) * (pts[2].y - pts[0].y) - (pts[2].x - pts[0].x) * (pts[1].y - pts[0].y);
            //int RemainingVertDirection = cross_z < 0.0f ? -1 : 1;

            //typedef struct{int YStart, Height; float X, Xit, LYPerc, LYPercIt, R, Rit;}SLineIt;

			if(cross_z < 0.0f)
			{
				line_it line_it_long = line_it_from_line(line_long.p0, line_long.p1);
				line_it line_it_top  = line_it_from_line(line_top.p0, line_top.p1);
				line_it line_it_bot  = line_it_from_line(line_bot.p0, line_bot.p1);

				int y=line_it_long.y_start;
				for(; y<line_it_long.y_start+line_it_top.height; y++)
				{
					const int x_left  = static_cast<int>(line_it_long.x);
					const int x_right = static_cast<int>(line_it_top.x);

					draw_hline_ctx ctx{y, x_left, x_right};
					draw_hline_fill_0xff(target, source_triangle, ctx);

					++line_it_long;
					++line_it_top;
				}

				for(; y<line_it_long.y_start+line_it_long.height; y++)
				{
					const int x_left  = static_cast<int>(line_it_long.x);
					const int x_right = static_cast<int>(line_it_bot.x);

					draw_hline_ctx ctx{y, x_left, x_right};
					draw_hline_fill_0xff(target, source_triangle, ctx);

					++line_it_long;
					++line_it_bot;
				}
			}
			else
			{
				line_it line_it_long = line_it_from_line(line_long.p0, line_long.p1);
				line_it line_it_top  = line_it_from_line(line_top.p0, line_top.p1);
				line_it line_it_bot  = line_it_from_line(line_bot.p0, line_bot.p1);

				int y=line_it_long.y_start;
				for(; y<line_it_long.y_start+line_it_top.height; y++)
				{
					const int x_right  = static_cast<int>(line_it_long.x);
					const int x_left = static_cast<int>(line_it_top.x);

					draw_hline_ctx ctx{y, x_left, x_right};
					draw_hline_fill_0xff(target, source_triangle, ctx);

					++line_it_long;
					++line_it_top;
				}

				for(; y<line_it_long.y_start+line_it_long.height; y++)
				{
					const int x_right  = static_cast<int>(line_it_long.x);
					const int x_left = static_cast<int>(line_it_bot.x);

					draw_hline_ctx ctx{y, x_left, x_right};
					draw_hline_fill_0xff(target, source_triangle, ctx);

					++line_it_long;
					++line_it_bot;
				}

			}

#if 0
            line_it its[3] = {
					line_it_from_line()
			};

            // create iterators
            union{struct{SLineIt Ll, Tl1, Tl2;}It;SLineIt Its[3];};

            for(int i=0; i<3; i++)
            {
                SLineIt& c = Its[i]; SLine& l = Lines[i];
                float Sx = l.p1x, Sy = l.p1y, Ex = l.p2x, Ey = l.p2y; // useful vars

                c.YStart = (int)ceil(Sy);
                c.Height = (int)ceil(Ey) - c.YStart; // set

                float Oh = 1.0f / (float)c.Height;
                c.Xit = (Ex - Sx) * Oh; // interpolation

                // calculate intencity
                c.LYPerc = (1.0f - (dot(Sx - Cx, Sy - Cy, XDir, YDir) / Dist)) * 40.0f;
                float PercEnd = (1.0f - (dot(Ex - Cx, Ey - Cy, XDir, YDir) / Dist)) * 40.0f;
                c.LYPercIt = (PercEnd - c.LYPerc) * Oh;

                float	SubPixel = ceil(Sy) - Sy;
                c.X = Sx + (c.Xit * SubPixel); // sub pixel
                c.LYPerc = c.LYPerc + (c.LYPercIt * SubPixel);
            }

#endif

			int p0_px_x_i = pts[0].x; // NOLINT
			int p0_px_y_i = pts[0].y; // NOLINT
			int p1_px_x_i = pts[1].x; // NOLINT
			int p1_px_y_i = pts[1].y; // NOLINT
			int p2_px_x_i = pts[2].x; // NOLINT
			int p2_px_y_i = pts[2].y; // NOLINT

			if(p0_px_x_i > 0 && p0_px_x_i < target.width && p0_px_y_i > 0 && p0_px_y_i < target.height)
			{
				target.data()[p0_px_x_i + p0_px_y_i * target.width] = 0xffffffff;
			}
			if(p1_px_x_i > 0 && p1_px_x_i < target.width && p1_px_y_i > 0 && p1_px_y_i < target.height)
			{
				target.data()[p1_px_x_i + p1_px_y_i * target.width] = 0xffffffff;
			}
			if(p2_px_x_i > 0 && p2_px_x_i < target.width && p2_px_y_i > 0 && p2_px_y_i < target.height)
			{
				target.data()[p2_px_x_i + p2_px_y_i * target.width] = 0xffffffff;
			}

		}

		void render_rasterize(context& context, render_target auto& target, const triangle_list auto& triangles, const camera auto& camera)
		{
			const float target_width_flt 	= static_cast<float>(target.width);  // NOLINT
			const float target_height_flt 	= static_cast<float>(target.height); // NOLINT
			const float aspect = target_width_flt / target_height_flt;
			const glm::mat4x4 perspective 	= glm::perspectiveLH(camera.fov, aspect, 0.1f, 100.0f);
			const glm::mat4x4 lookat 		= glm::lookAtLH(glm::vec3{camera.pos.x, camera.pos.y, camera.pos.z}, glm::vec3{camera.lookat.x, camera.lookat.y, camera.lookat.z}, glm::vec3{camera.up.x, camera.up.y, camera.up.z});

			const glm::mat4x4 projview = lookat * perspective;

			for(const triangle auto& tri : triangles)
			{
				const glm::vec4 p0 = {tri.p0.x, tri.p0.y, tri.p0.z, 1};
				const glm::vec4 p1 = {tri.p1.x, tri.p1.y, tri.p1.z, 1};
				const glm::vec4 p2 = {tri.p2.x, tri.p2.y, tri.p2.z, 1};

				const glm::vec4 p0_projview = projview * p0;
				const glm::vec4 p1_projview = projview * p1;
				const glm::vec4 p2_projview = projview * p2;

				assert(p0_projview.w != 0.0f);

                std::array<grh::vec3, 3> pts; // NOLINT

				pts[0].x = ((p0_projview.x / p0_projview.w) * 0.5f + 0.5f) * target_width_flt;
				pts[0].y = ((p0_projview.y / p0_projview.w) * 0.5f + 0.5f) * target_height_flt;
				pts[0].z = (p0_projview.z / p0_projview.w);

				pts[1].x = ((p1_projview.x / p1_projview.w) * 0.5f + 0.5f) * target_width_flt;
				pts[1].y = ((p1_projview.y / p1_projview.w) * 0.5f + 0.5f) * target_height_flt;
				pts[1].z = (p1_projview.z / p1_projview.w);

				pts[2].x = ((p2_projview.x / p2_projview.w) * 0.5f + 0.5f) * target_width_flt;
				pts[2].y = ((p2_projview.y / p2_projview.w) * 0.5f + 0.5f) * target_height_flt;
				pts[2].z = (p2_projview.z / p2_projview.w);

				if(pts[0].z > 0.0f || pts[1].z > 0.0f || pts[2].z > 0.0f)
				{
					draw_triangle(target, tri, pts);
				}
			}
		}

		inline detail::impl& context_detail(gr::context& ctx)
		{
			if(ctx.impl == nullptr)
			{
				ctx.impl = std::make_unique<detail::impl>();
			}
			return *ctx.impl;
		}
	}

	context create_context()
	{
		return context{};
	}

	void clear(render_target auto& target)
	{
		std::span<uint32_t> target_rgb(target.data(), target.width * target.height);
		memset(target_rgb.data(), 0, target_rgb.size_bytes());
	}

	void render(context& context, render_target auto& target, const triangle_list auto& triangles, const camera auto& camera)
	{
		detail::render_rasterize(context, target, triangles, camera);
	}

	void render(render_target auto& target, const triangle_list auto& triangles, const camera auto& camera)
	{
		auto ctx = create_context();
		render(ctx, target, triangles, camera);
	}

	namespace helpers
	{
		inline managed_render_target create_render_target(size_t num_pixels_width, size_t num_pixels_height)
		{
			managed_render_target target;
			target.width = num_pixels_width;
			target.height = num_pixels_height;
			target.rgbd_buffer 	= std::unique_ptr<uint32_t>(new(std::align_val_t(32)) uint32_t[target.width * target.height]);
			return target;
		}

		cam lookat(const vector3 auto& pos, const vector3 auto& lookat, const vector3 auto& up, const std::floating_point auto& fov)
		{
			return cam{
				.pos = {pos.x, pos.y, pos.z},
				.lookat = {lookat.x, lookat.y, lookat.z},
				.up = {up.x, up.y, up.z},
				.fov = fov
			};
		}
	}
}



#endif //CPURAYTRACER_GT_HPP
