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
		{*v.begin()} -> triangle;
		{*v.end()} -> triangle;
	};

	template<triangle_list T>
	using triangle_from_list_t = decltype(*std::declval<T>().begin());

	template<typename T>
	concept camera = requires(T v)
	{
		{v.pos} 			-> vector3;
		{v.lookat} 			-> vector3;
		{v.up} 				-> vector3;
		{v.fov} 			-> floating_point;
	};

	template<typename T>
	concept draw_horizontal_line_ctx = requires(T v)
	{
		{v.buffer_width} 	-> unsigned_integral;
		{v.buffer_height} 	-> unsigned_integral;
		{v.px_y} 			-> unsigned_integral;
		{v.px_x_from} 		-> unsigned_integral;
		{v.px_x_to} 		-> unsigned_integral;
	};
}

namespace gr::helpers
{
	struct vec3
	{
		float x,y,z;
	};
	static_assert(vector3<vec3>);

	struct tri
	{
		vec3 p0, p1, p2;
	};
	static_assert(triangle<tri>);

	struct cam
	{
		vec3 pos, lookat, up;
		float fov;
	};
	static_assert(camera<cam>);
}
namespace grh = gr::helpers;

namespace gr
{

	struct draw_hline_ctx
	{
		uint32_t buffer_width;
		uint32_t buffer_height;
		uint32_t px_y, px_x_from, px_x_to;
	};

	namespace detail::concepts
	{
		struct draw_horizontal_line_ctx_example
		{
			size_t buffer_width, buffer_height, px_y, px_x_from, px_x_to;
		};
		static_assert(draw_horizontal_line_ctx<draw_horizontal_line_ctx_example>);
	}

	template<typename FuncT, typename TriangleType>
	concept draw_horizontal_line_function = requires(FuncT v)
	{
		{v(std::declval<TriangleType>(), std::declval<draw_hline_ctx>())};
	};

	template<triangle_list TriangleList>
	constexpr void 	render(const TriangleList& triangles, const camera auto& camera, draw_horizontal_line_function<triangle_from_list_t<TriangleList>> auto&& draw_hline_function, unsigned_integral auto frame_width, unsigned_integral auto frame_height);
}

namespace gr::helpers
{
	constexpr cam					lookat(const vector3 auto& pos, const vector3 auto& lookat, const vector3 auto& up, const std::floating_point auto& fov = 2.0f);
}


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
			const float one_over_height_ceiled 	= height_ceiled != 0.0f ? (1.0f / height_ceiled) : 0.0f;
			const float sub_pixel 				= y_start_ceiled - p0.y;
			//assert(height_ceiled != 0.0f); // this is going to be a division over 0 ! // TODO: handle this to avoid NaN

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

		constexpr inline void draw_hline_fill_0xff(const triangle auto& source_triangle, const draw_hline_ctx& ctx)
		{
			assert(ctx.px_x_to 		>= ctx.px_x_from);
			assert(ctx.px_y 		< ctx.buffer_height);
			assert(ctx.px_x_from 	< ctx.buffer_width);
			assert(ctx.px_x_to 		< ctx.buffer_width);
			//std::span<uint32_t> target_data{target.data(), ctx.buffer_width * ctx.buffer_height};
			//auto begin = std::next(target_data.begin(), ctx.x_from + ctx.y * ctx.buffer_width);
			//auto end = std::next(begin, ctx.x_to - ctx.x_from);
			//std::fill(begin, end, 0xffffffff);
		}

		template<triangle TriangleType>
		constexpr void draw_triangle(const TriangleType& source_triangle, std::array<grh::vec3, 3>& pts, draw_horizontal_line_function<TriangleType> auto&& draw_hline_function, unsigned_integral auto frame_width, unsigned_integral auto frame_height)
		{
            struct line {const grh::vec3& p0, &p1;};

            std::sort(pts.begin(), pts.end(), [](const grh::vec3& a, const grh::vec3& b){return a.y < b.y;});

            // take lines
            const line line_long  = {pts[0], pts[2]};
            const line line_top   = {pts[0], pts[1]};
            const line line_bot   = {pts[1], pts[2]};

            // check whether the long line is on the left or right
            float cross_z = (pts[1].x - pts[0].x) * (pts[2].y - pts[0].y) - (pts[2].x - pts[0].x) * (pts[1].y - pts[0].y);

			draw_hline_ctx ctx; // NOLINT
			ctx.buffer_width = static_cast<uint32_t>(frame_width);
			ctx.buffer_height = static_cast<uint32_t>(frame_height);

			if(cross_z > 0.0f)
			{
				line_it line_it_long = line_it_from_line(line_long.p0, line_long.p1);
				line_it line_it_top  = line_it_from_line(line_top.p0, line_top.p1);
				line_it line_it_bot  = line_it_from_line(line_bot.p0, line_bot.p1);

				int y=line_it_long.y_start;
				for(; y<line_it_long.y_start+line_it_top.height; y++)
				{
					ctx.px_y 		= y;
					ctx.px_x_from 	= static_cast<int>(line_it_long.x);
					ctx.px_x_to 	= static_cast<int>(line_it_top.x);

					assert(ctx.px_x_to 		>= ctx.px_x_from);
					assert(ctx.px_y 		< ctx.buffer_height);
					assert(ctx.px_x_from 	< ctx.buffer_width);
					assert(ctx.px_x_to 		< ctx.buffer_width);
					draw_hline_function(source_triangle, ctx);

					++line_it_long;
					++line_it_top;
				}

				for(; y<line_it_long.y_start+line_it_long.height; y++)
				{
					ctx.px_y 		= y;
					ctx.px_x_from 	= static_cast<int>(line_it_long.x);
					ctx.px_x_to 	= static_cast<int>(line_it_bot.x);

					assert(ctx.px_x_to 		>= ctx.px_x_from);
					assert(ctx.px_y 		< ctx.buffer_height);
					assert(ctx.px_x_from 	< ctx.buffer_width);
					assert(ctx.px_x_to 		< ctx.buffer_width);
					draw_hline_function(source_triangle, ctx);

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
					ctx.px_y 		= y;
					ctx.px_x_from 	= static_cast<int>(line_it_top.x);
					ctx.px_x_to 	= static_cast<int>(line_it_long.x);

					assert(ctx.px_x_to 		>= ctx.px_x_from);
					assert(ctx.px_y 		< ctx.buffer_height);
					assert(ctx.px_x_from 	< ctx.buffer_width);
					assert(ctx.px_x_to 		< ctx.buffer_width);
					draw_hline_function(source_triangle, ctx);

					++line_it_long;
					++line_it_top;
				}

				for(; y<line_it_long.y_start+line_it_long.height; y++)
				{
					ctx.px_y 		= y;
					ctx.px_x_from 	= static_cast<int>(line_it_bot.x);
					ctx.px_x_to 	= static_cast<int>(line_it_long.x);

					assert(ctx.px_x_to 		>= ctx.px_x_from);
					assert(ctx.px_y 		< ctx.buffer_height);
					assert(ctx.px_x_from 	< ctx.buffer_width);
					assert(ctx.px_x_to 		< ctx.buffer_width);
					draw_hline_function(source_triangle, ctx);

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

#if 0
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
#endif
		}

		template<triangle_list TriangleList>
		constexpr void render_rasterize(const TriangleList& triangles, const camera auto& camera, draw_horizontal_line_function<triangle_from_list_t<TriangleList>> auto&& draw_hline_function, unsigned_integral auto frame_width, unsigned_integral auto frame_height)
		{
			const float target_width_flt 	= static_cast<float>(frame_width);  // NOLINT
			const float target_height_flt 	= static_cast<float>(frame_height); // NOLINT
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
					draw_triangle(tri, pts, draw_hline_function, frame_width, frame_height);
				}
			}
		}
	}

	template<triangle_list TriangleList>
	constexpr void 	render(const TriangleList& triangles, const camera auto& camera, draw_horizontal_line_function<triangle_from_list_t<TriangleList>> auto&& draw_hline_function, unsigned_integral auto frame_width, unsigned_integral auto frame_height)
	{
		detail::render_rasterize(triangles, camera, draw_hline_function, frame_width, frame_height);
	}

	namespace helpers
	{
		constexpr cam lookat(const vector3 auto& pos, const vector3 auto& lookat, const vector3 auto& up, const std::floating_point auto& fov)
		{
			return cam{
				.pos = {pos.x, pos.y, pos.z},
				.lookat = {lookat.x, lookat.y, lookat.z},
				.up = {up.x, up.y, up.z},
				.fov = fov
			};
		}
	}

#if 0
	namespace tests
	{
		struct test_draw
		{
			constexpr void operator()(const grh::tri& source_triangle, const gr::draw_hline_ctx& ctx)
			{

			}
		};


		constexpr inline void test_draw_func(const triangle auto& source_triangle, const draw_horizontal_line_ctx auto& ctx)
		{
		}

		constexpr bool test()
		{
			auto camera 						= grh::lookat(grh::vec3{0,0,0}, grh::vec3{0,0,1}, grh::vec3{0,1,0}, 2.0f);
			std::array<grh::tri, 1> triangles 	= {grh::tri{{0,0,5}, {2,1,5}, {0,2,5}}};

			test_draw testdraw;

			detail::render_rasterize(triangles, camera, testdraw, 100u, 100u);
			return true;
		}

		//constexpr bool test()
		//{
		//	auto camera 						= grh::lookat(grh::vec3{0,0,0}, grh::vec3{0,0,1}, grh::vec3{0,1,0}, 2.0f);
		//	std::array<grh::tri, 1> triangles 	= {grh::tri{{0,0,5}, {2,1,5}, {0,2,5}}};
		//	test_draw testdraw;
		//	detail::render_rasterize(triangles, camera, [](const grh::tri& source_triangle, const gr::draw_hline_ctx& ctx){}, 100u, 100u);
		//	return true;
		//}

		static_assert(test());

	}
#endif
}



#endif //CPURAYTRACER_GT_HPP
