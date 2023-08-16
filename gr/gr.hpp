#ifndef CPURAYTRACER_GT_HPP
#define CPURAYTRACER_GT_HPP

#include <array>
#include <cstdint>
#include <memory>
#include <span>

namespace gr
{
	template<typename T>
	concept floating_point = std::floating_point<std::decay_t<T>>;

	template<typename T>
	concept unsigned_integral = std::unsigned_integral<std::decay_t<T>>;

	template<typename T>
	concept unsigned_int_32 = std::same_as<uint32_t, std::decay_t<T>>;

	template<typename T>
	concept vector2 = requires(T v)
	{
		{v.x} -> floating_point;
		{v.y} -> floating_point;
	};

	template<typename T>
	concept vector3 = requires(T v)
	{
		{v.x} -> floating_point;
		{v.y} -> floating_point;
		{v.z} -> floating_point;
	};

	template<typename T>
	concept triangle_by_fields = requires(T v)
	{
		{v.p0} -> vector3;
		{v.p1} -> vector3;
		{v.p2} -> vector3;
	};

	template<typename T>
	concept triangle_by_indices = requires(T v)
	{
		{v[0]} -> vector3;
		{v[1]} -> vector3;
		{v[2]} -> vector3;
	};

	template<typename T>
	concept triangle = triangle_by_fields<T> || triangle_by_indices<T>;

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
	struct vec2
	{
		float x,y;
	};
	static_assert(vector2<vec2>);

	struct vec3
	{
		float x,y,z;
	};
	static_assert(vector3<vec3>);

	struct vec4
	{
		float x,y,z,w;
	};

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

	using mat4x4 = std::array<std::array<float, 4>, 4>;
}
namespace grh = gr::helpers;

namespace gr
{

	struct draw_hline_ctx
	{
		uint32_t buffer_width;
		uint32_t buffer_height;
		uint32_t px_y, px_x_from, line_length_px;
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

	template<typename IteratorCollection, triangle_list TriangleList>
	constexpr void 	render(const TriangleList& triangles, const camera auto& camera, draw_horizontal_line_function<triangle_from_list_t<TriangleList>> auto&& draw_hline_function, unsigned_integral auto frame_width, unsigned_integral auto frame_height);

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
#if __has_include(<gcem.hpp>)
#include <gcem.hpp>
#define HAS_GCEM
#endif

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

		template<typename ...Iterators>
		struct iterator_list
		{
		};

		struct iterator_collection
		{
			//void init(const vector3 auto& p0, const vector3 auto& p1, float one_over_height_ceiled)
			//{
			//
			//}

			template<typename Cb>
			void for_each(Cb cb)
			{

			}
		};

		struct z_iterator
		{
			explicit z_iterator(const vector3 auto& v) : value(v.z){}
			float value;
		};

		template<typename IteratorCollection>
        struct line_it
        {
			int y_start, height;

			float x_it, x;
			float z_it, z;

			IteratorCollection iterator_collection;

			line_it& operator ++()
			{
				x += x_it;
				z += z_it;
				iterator_collection.for_each([&](auto& v) { v.value += v.value_it; });
				return *this;
			}
        };

		template<typename VecA, typename VecB>
		using smallest_vec3_t = std::conditional_t<sizeof(VecA) < sizeof(VecB), VecA, VecB>;

		template<typename IteratorCollection>
		line_it<IteratorCollection> line_it_from_line(const vector3 auto& p0, const vector3 auto& p1)
        {
			const float y_start_ceiled 			= std::ceil(p0.y);
			const float height_ceiled 			= std::ceil(p1.y) - y_start_ceiled;
			const float one_over_height_ceiled 	= height_ceiled != 0.0f ? (1.0f / height_ceiled) : 0.0f;
			const float sub_pixel 				= y_start_ceiled - p0.y;
			//assert(height_ceiled != 0.0f); // this is going to be a division over 0 ! // TODO: handle this to avoid NaN

            line_it<IteratorCollection> c {
                    .y_start    = static_cast<int>(y_start_ceiled),
                    .height     = static_cast<int>(height_ceiled),
					.x_it 		= (p1.x - p0.x) * one_over_height_ceiled,
					.x			= p0.x + (c.x_it * sub_pixel),
					.z_it 		= (p1.z - p0.z) * one_over_height_ceiled,
					.z			= p0.z + (c.z_it * sub_pixel)
			};

			c.iterator_collection.for_each([&](auto& v)
			{
				v.value_it 	= (v.field(p1) - v.field(p0)) * one_over_height_ceiled;
				v.value		= v.field(p0) + (v.value_it * sub_pixel);
			});

			return c;
		}

		template<typename IteratorCollection>
		constexpr inline int y_pos_at_intersection(int current_ypos, const line_it<IteratorCollection>& a, const line_it<IteratorCollection>& b)
		{
			if(b.x_it - a.x_it == 0.0f)
			{
				return (current_ypos-1);
			}
			//assert((b.x_it - a.x_it) != 0.0f);
			int lowest_y = std::min(a.y_start + a.height, b.y_start + b.height);
			int calculatedIterationsLeft = static_cast<int>(std::floor((a.x - b.x) / (b.x_it - a.x_it)));
			int y_limit = current_ypos + calculatedIterationsLeft;
			return std::min(y_limit, lowest_y);
		}

		inline constexpr void check_context_validity(draw_hline_ctx& ctx)
		{
#if 0
			int to = ctx.px_x_from + ctx.line_length_px;
			assert(to 				>= ctx.px_x_from);
			assert(ctx.px_y 		< ctx.buffer_height);
			assert(ctx.px_x_from 	< ctx.buffer_width);
			assert(ctx.px_x_to 		< ctx.buffer_width);
#else
			ctx.px_y 			= std::min(ctx.px_y, (ctx.buffer_height-1));
			ctx.px_x_from 		= std::min(ctx.px_x_from, (ctx.buffer_width-1));
			ctx.line_length_px 	= (ctx.px_x_from + ctx.line_length_px) > (ctx.buffer_width-1) ? ((ctx.buffer_width-1) - ctx.px_x_from) : ctx.line_length_px;
#endif
		}

		/// no bounds checking here!
		template<typename IteratorCollection, triangle TriangleType>
		constexpr void draw_triangle_unsafe(const TriangleType& source_triangle, std::array<grh::vec3, 3>& pts_screen_space, draw_horizontal_line_function<TriangleType> auto&& draw_hline_function, unsigned_integral auto frame_width, unsigned_integral auto frame_height)
		{
            struct line {const grh::vec3& p0, &p1;};

            std::sort(pts_screen_space.begin(), pts_screen_space.end(), [](const grh::vec3& a, const grh::vec3& b){return a.y < b.y;});

            // take lines
            const line line_long  = {pts_screen_space[0], pts_screen_space[2]};
            const line line_top   = {pts_screen_space[0], pts_screen_space[1]};
            const line line_bot   = {pts_screen_space[1], pts_screen_space[2]};

            // check whether the long line is on the left or right
            float cross_z = (pts_screen_space[1].x - pts_screen_space[0].x) * (pts_screen_space[2].y - pts_screen_space[0].y) - (pts_screen_space[2].x - pts_screen_space[0].x) * (pts_screen_space[1].y - pts_screen_space[0].y);

			draw_hline_ctx ctx; // NOLINT
			ctx.buffer_width = static_cast<uint32_t>(frame_width);
			ctx.buffer_height = static_cast<uint32_t>(frame_height);

			if(cross_z > 0.0f)
			{
				line_it<IteratorCollection> line_it_long = line_it_from_line<IteratorCollection>(line_long.p0, line_long.p1);
				line_it<IteratorCollection> line_it_top  = line_it_from_line<IteratorCollection>(line_top.p0, line_top.p1);
				line_it<IteratorCollection> line_it_bot  = line_it_from_line<IteratorCollection>(line_bot.p0, line_bot.p1);

				int y=line_it_long.y_start;
				for(; y<line_it_long.y_start+line_it_top.height; y++)
				{
					ctx.px_y 			= y;
					ctx.px_x_from 		= static_cast<int>(line_it_long.x);
					ctx.line_length_px 	= static_cast<int>(line_it_top.x) - ctx.px_x_from;

					check_context_validity(ctx);
					draw_hline_function(source_triangle, ctx);

					++line_it_long;
					++line_it_top;
				}

				const int yLimit = y_pos_at_intersection(y, line_it_long, line_it_bot);
				for(; y<yLimit; y++)
				{
					ctx.px_y 			= y;
					ctx.px_x_from 		= static_cast<int>(line_it_long.x);
					ctx.line_length_px 	= static_cast<int>(line_it_bot.x) - ctx.px_x_from;

					check_context_validity(ctx);
					draw_hline_function(source_triangle, ctx);

					++line_it_long;
					++line_it_bot;
				}
			}
			else
			{
				line_it line_it_long = line_it_from_line<IteratorCollection>(line_long.p0, line_long.p1);
				line_it line_it_top  = line_it_from_line<IteratorCollection>(line_top.p0, line_top.p1);
				line_it line_it_bot  = line_it_from_line<IteratorCollection>(line_bot.p0, line_bot.p1);

				int y=line_it_long.y_start;
				for(; y<line_it_long.y_start+line_it_top.height; y++)
				{
					ctx.px_y 			= y;
					ctx.px_x_from 		= static_cast<int>(line_it_top.x);
					ctx.line_length_px 	= static_cast<int>(line_it_long.x) - ctx.px_x_from;

					check_context_validity(ctx);
					draw_hline_function(source_triangle, ctx);

					++line_it_long;
					++line_it_top;
				}

				const int yLimit = y_pos_at_intersection(y, line_it_long, line_it_bot);
				for(; y<yLimit; y++)
				{
					ctx.px_y 			= y;
					ctx.px_x_from 		= static_cast<int>(line_it_bot.x);
					ctx.line_length_px	= static_cast<int>(line_it_long.x) - ctx.px_x_from;

					check_context_validity(ctx);
					draw_hline_function(source_triangle, ctx);

					++line_it_long;
					++line_it_bot;
				}
			}
		}

		/// does not check for 0 division
		template<vector3 Vec3Type>
		constexpr Vec3Type normalize(const Vec3Type& v)
		{
			const auto l = static_cast<std::decay_t<decltype(v.x)>>(1) / std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
			return {v.x / l, v.y / l, v.z / l};
		}

		/// does not check for 0 division
		template<vector3 Vec3Type>
		constexpr Vec3Type direction_to(const Vec3Type& from, const Vec3Type& to)
		{
			return normalize(Vec3Type{to.x - from.x, to.y - from.y, to.z - from.z});
		}

		/// does not check for 0 division
		constexpr auto length(const vector3 auto& vec)
		{
			return std::sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
		}

		constexpr grh::vec3 cross_xyz(const vector3 auto& a, const vector3 auto& b)
		{
			return {a.b * b.z - b.b * a.z, a.z * b.a - b.z * a.a, a.a * b.b - b.a * a.b};
		}

		constexpr floating_point auto dot_xyz(const vector3 auto& a, const vector3 auto& b)
		{
			return a.x * b.x + a.y * b.y + a.z * b.z;
		}

		template<vector3 Vec3AT, vector3 Vec3BT>
		constexpr smallest_vec3_t<Vec3AT, Vec3BT> sub_xyz(const Vec3AT& a, const Vec3BT& b)
		{
			return {a.x - b.x, a.y - b.y, a.z - b.z};
		}

		template<vector3 Vec3AT, vector3 Vec3BT>
		constexpr smallest_vec3_t<Vec3AT, Vec3BT> add_xyz(const Vec3AT& a, const Vec3BT& b)
		{
			return {a.x + b.x, a.y + b.y, a.z + b.z};
		}

		template<vector3 Vec3AT, vector3 Vec3BT>
		constexpr smallest_vec3_t<Vec3AT, Vec3BT> mul_xyz(const Vec3AT& a, const Vec3BT& b)
		{
			return {a.x * b.x, a.y * b.y, a.z * b.z};
		}

		template<vector3 Vec3T>
		constexpr Vec3T mul_xyz(const Vec3T& a, float b)
		{
			return {a.x * b, a.y * b, a.z * b};
		}

		constexpr grh::mat4x4 mul(const grh::mat4x4& m1, const grh::mat4x4& m2)
		{
			grh::mat4x4 result;
			for(size_t k=0; k<4; k++) for(size_t i=0; i<4; i++)
				{
					result[k][i] = m1[0][i] * m2[k][0] + m1[1][i] * m2[k][1] + m1[2][i] * m2[k][2] + m1[3][i] * m2[k][3];
				}
			return result;
		}

		constexpr grh::vec4 mul(const grh::mat4x4& m1, const grh::vec4& m2)
		{
			grh::vec4 result = {};
			result.x = m1[0][0] * m2.x + m1[1][0] * m2.y + m1[2][0] * m2.z + m1[3][0] * m2.w;
			result.y = m1[0][1] * m2.x + m1[1][1] * m2.y + m1[2][1] * m2.z + m1[3][1] * m2.w;
			result.z = m1[0][2] * m2.x + m1[1][2] * m2.y + m1[2][2] * m2.z + m1[3][2] * m2.w;
			result.w = m1[0][3] * m2.x + m1[1][3] * m2.y + m1[2][3] * m2.z + m1[3][3] * m2.w;
			return result;
		}

		template<size_t Index>
		constexpr auto& get_tri_pt(triangle_by_fields auto& tri) requires (Index < 3)
		{
			if 		constexpr( Index == 0)		return tri.p0;
			else if constexpr( Index == 1)		return tri.p1;
			else if constexpr( Index == 2)		return tri.p2;
			else								return tri.p0;
		}

		template<size_t Index>
		constexpr auto& get_tri_pt(triangle_by_indices auto& tri) requires (Index < 3)
		{
			return tri[Index];
		}

		constexpr auto& get_tri_p0(triangle auto& tri) { return get_tri_pt<0>(tri); }
		constexpr auto& get_tri_p1(triangle auto& tri) { return get_tri_pt<1>(tri); }
		constexpr auto& get_tri_p2(triangle auto& tri) { return get_tri_pt<2>(tri); }

		constexpr floating_point auto intersect(const vector3 auto& ray_origin, const vector3 auto& ray_dir, const vector3 auto& plane_pos, const vector3 auto& plane_normal)
		{
			const floating_point auto denom = (plane_normal.x * ray_dir.x) + (plane_normal.y * ray_dir.y) + (plane_normal.z * ray_dir.z);
			if ((denom*denom) > static_cast<decltype(denom)>(0.0001 * 0.0001)) // your favorite epsilon
			{
				const vector3 auto 			d = sub_xyz(plane_pos, ray_origin);
				const floating_point auto 	d_dot = dot_xyz(d, plane_normal);
				return d_dot / denom;
			}
			return static_cast<decltype(denom)>(-1.0f);
		}

		constexpr int clip_triangle(const vector3 auto& plane_pos, const vector3 auto& plane_normal, triangle auto& tri_in_out, triangle auto& tri_extra_out)
		{
			const vector3 auto& a = get_tri_p0(tri_in_out);
			const vector3 auto& b = get_tri_p1(tri_in_out);
			const vector3 auto& c = get_tri_p2(tri_in_out);

			const vector3 auto ab = sub_xyz(b, a);
			const vector3 auto bc = sub_xyz(c, b);
			const vector3 auto ca = sub_xyz(a, c);

			const floating_point auto ab_len = length(ab);
			const floating_point auto bc_len = length(bc);
			const floating_point auto ca_len = length(ca);

			const floating_point auto ab_len_inv = 1.0f / ab_len;
			const floating_point auto bc_len_inv = 1.0f / bc_len;
			const floating_point auto ca_len_inv = 1.0f / ca_len;

			const vector3 auto ab_dir = mul_xyz(ab, ab_len_inv); // TODO: This can be precalculated
			const vector3 auto bc_dir = mul_xyz(bc, bc_len_inv); // TODO: This can be precalculated
			const vector3 auto ca_dir = mul_xyz(ca, ca_len_inv); // TODO: This can be precalculated

			const floating_point auto a_to_b_t = intersect(a, ab_dir, plane_pos, plane_normal);
			const floating_point auto b_to_c_t = intersect(b, bc_dir, plane_pos, plane_normal);
			const floating_point auto c_to_a_t = intersect(c, ca_dir, plane_pos, plane_normal);

			const bool intersects_a_to_b = (a_to_b_t > 0 && a_to_b_t < ab_len);
			const bool intersects_b_to_c = (b_to_c_t > 0 && b_to_c_t < bc_len);
			const bool intersects_c_to_a = (c_to_a_t > 0 && c_to_a_t < ca_len);

			const vector3 auto intersection_a_to_b = add_xyz(a, mul_xyz(ab_dir, a_to_b_t));
			const vector3 auto intersection_b_to_c = add_xyz(b, mul_xyz(bc_dir, b_to_c_t));
			const vector3 auto intersection_c_to_a = add_xyz(c, mul_xyz(ca_dir, c_to_a_t));

			const vector3 auto& triangle_tip  = (intersects_a_to_b && intersects_c_to_a) ? a : ((intersects_a_to_b && intersects_b_to_c) ? b : c);
			const vector3 auto& any_other_tip = (intersects_a_to_b && intersects_c_to_a) ? b : ((intersects_a_to_b && intersects_b_to_c) ? c : a);

			if(intersects_a_to_b || intersects_b_to_c || intersects_c_to_a)
			{
				const vector3 auto	tip_to_tip			= sub_xyz(triangle_tip, any_other_tip);
				const bool 			should_cut_to_quad 	= (tip_to_tip.x*plane_normal.x + tip_to_tip.y*plane_normal.y + tip_to_tip.z*plane_normal.z) < 0.0f;
				const std::decay_t<decltype(a)> points[] = {a, b, c, intersection_a_to_b, intersection_b_to_c, intersection_c_to_a};

				struct // NOLINT
				{
					std::uint_fast8_t indices[4]; // corresponds to the index of 'points'. so max num is 5  ( points.size() - 1 )
					std::uint_fast8_t num_indices = 2;
				} constexpr s_point_indices[] =
						{
								{ }, 					// 	0, triangle has not intersection after all ?
								{ { 2, 5, 4}, 3 }, 		// 	1, intersection between c->a and c->b, cut out triangle
								{ { 0, 3, 5}, 3 }, 		// 	2, intersection between a->b and a->c, cut out triangle
								{ { 1, 3, 4}, 3 }, 		// 	3, intersection between b->a and b->c, cut out triangle
								{ }, 					// 	4, triangle has not intersection after all ?
								{ { 0, 1, 4, 5}, 4 }, 	// 	5, intersection between c->a and c->b, cut out quad
								{ { 1, 2, 5, 3}, 4 }, 	// 	6, intersection between a->b and a->c, cut out quad
								{ { 2, 0, 3, 4}, 4 }, 	// 	7, intersection between b->a and b->c, cut out quad
						};

				const uint state = ((!intersects_a_to_b ? 0b0001u : 0u) | (!intersects_b_to_c ? 0b0010u : 0u) | (!intersects_c_to_a ? 0b0011u : 0u)) + (should_cut_to_quad ? 0b0100u : 0u);
				const auto& indices = s_point_indices[state];

				uint num_tris = indices.num_indices - 2;
				assert(num_tris == 1 || num_tris == 2);
				get_tri_p0(tri_in_out) = points[indices.indices[((0<<1u)+0u)&0b11u]];
				get_tri_p1(tri_in_out) = points[indices.indices[((0<<1u)+1u)&0b11u]];
				get_tri_p2(tri_in_out) = points[indices.indices[((0<<1u)+2u)&0b11u]];

				if(num_tris == 2)
				{
					get_tri_p0(tri_extra_out) = points[indices.indices[((1<<1u)+0u)&0b11u]];
					get_tri_p1(tri_extra_out) = points[indices.indices[((1<<1u)+1u)&0b11u]];
					get_tri_p2(tri_extra_out) = points[indices.indices[((1<<1u)+2u)&0b11u]];
					return 1;
				}
			}
			else
			{
				return dot_xyz(sub_xyz(c, plane_pos), plane_normal) > 0 ? 0 : -1;
			}
			return 0;
		}

		template<typename IteratorCollection, triangle TriangleType>
		constexpr void draw_triangle(const TriangleType& source_triangle, std::array<grh::vec3, 3>& pts_screen_space, draw_horizontal_line_function<TriangleType> auto&& draw_hline_function, unsigned_integral auto frame_width, unsigned_integral auto frame_height,
									 unsigned_integral auto viewport_x_start, unsigned_integral auto viewport_y_start, unsigned_integral auto viewport_x_end, unsigned_integral auto viewport_y_end)
		{
			// clipping on edges

			std::array<std::array<grh::vec3, 3>, (2*2*2*2)+1> 	clipped_tris; // NOLINT
			std::uint_fast8_t 									clipped_tris_num = 0;

			// first just add the main triangle
			clipped_tris[clipped_tris_num++] = pts_screen_space;

			// clipping planes
			// top clipping plane
			{
				const grh::vec3 n = {0, -1, 0};
				const grh::vec3 o = {0, static_cast<float>(viewport_y_end), 0};

				for(std::uint_fast8_t i=clipped_tris_num; i--;)
				{
					std::array<grh::vec3, 3>& tri = clipped_tris[i];

					const bool 					in_screen[3] 		= {tri[0].y < viewport_y_end, tri[1].y < viewport_y_end, tri[2].y < viewport_y_end};
					const std::uint_fast8_t 	num_pts_in_screen 	= in_screen[0] + in_screen[1] + in_screen[2];

					if(num_pts_in_screen == 0) // not in screen. delete
					{
						std::swap(clipped_tris[i], clipped_tris[clipped_tris_num-1]);
						clipped_tris_num--;
					}
					else if(num_pts_in_screen != 3) // otherwise nothing to clip. leave it
					{
						clipped_tris_num += clip_triangle(o, n, tri, clipped_tris[clipped_tris_num]);
					}
				}
			}

			// bot clipping plane
			{
				const grh::vec3 n = {0, 1, 0};
				const grh::vec3 o = {0, static_cast<float>(viewport_y_start), 0};

				for(std::uint_fast8_t i=clipped_tris_num; i--;)
				{
					std::array<grh::vec3, 3>& tri = clipped_tris[i];

					const bool 					in_screen[3] 		= {tri[0].y > viewport_y_start, tri[1].y > viewport_y_start, tri[2].y > viewport_y_start};
					const std::uint_fast8_t 	num_pts_in_screen 	= in_screen[0] + in_screen[1] + in_screen[2];

					if(num_pts_in_screen == 0) // not in screen. delete
					{
						std::swap(clipped_tris[i], clipped_tris[clipped_tris_num-1]);
						clipped_tris_num--;
					}
					else if(num_pts_in_screen != 3) // otherwise nothing to clip. leave it
					{
						clipped_tris_num += clip_triangle(o, n, tri, clipped_tris[clipped_tris_num]);
					}
				}
			}

			// left clipping plane
			{
				const grh::vec3 n = {1, 0, 0};
				const grh::vec3 o = {static_cast<float>(viewport_x_start), 0, 0};

				for(std::uint_fast8_t i=clipped_tris_num; i--;)
				{
					std::array<grh::vec3, 3>& tri = clipped_tris[i];

					const bool 					in_screen[3] 		= {tri[0].x > viewport_x_start, tri[1].x > viewport_x_start, tri[2].x > viewport_x_start};
					const std::uint_fast8_t 	num_pts_in_screen 	= in_screen[0] + in_screen[1] + in_screen[2];

					if(num_pts_in_screen == 0) // not in screen. delete
					{
						std::swap(clipped_tris[i], clipped_tris[clipped_tris_num-1]);
						clipped_tris_num--;
					}
					else if(num_pts_in_screen != 3) // otherwise nothing to clip. leave it
					{
						clipped_tris_num += clip_triangle(o, n, tri, clipped_tris[clipped_tris_num]);
					}
				}
			}

			// right clipping plane
			{
				const grh::vec3 n = {-1, 0, 0};
				const grh::vec3 o = {static_cast<float>(viewport_x_end), 0, 0};

				for(std::uint_fast8_t i=clipped_tris_num; i--;)
				{
					std::array<grh::vec3, 3>& tri = clipped_tris[i];

					const bool 					in_screen[3] 		= {tri[0].x < viewport_x_end, tri[1].x < viewport_x_end, tri[2].x < viewport_x_end};
					const std::uint_fast8_t 	num_pts_in_screen 	= in_screen[0] + in_screen[1] + in_screen[2];

					if(num_pts_in_screen == 0) // not in screen. delete
					{
						std::swap(clipped_tris[i], clipped_tris[clipped_tris_num-1]);
						clipped_tris_num--;
					}
					else if(num_pts_in_screen != 3) // otherwise nothing to clip. leave it
					{
						clipped_tris_num += clip_triangle(o, n, tri, clipped_tris[clipped_tris_num]);
					}
				}
			}

			// draw
			for(std::uint_fast8_t i=0; i<clipped_tris_num; i++)
			{
				draw_triangle_unsafe<IteratorCollection>(source_triangle, clipped_tris[i], draw_hline_function, frame_width, frame_height);
			}
		}

		template<typename IteratorCollection, triangle TriangleType>
		constexpr void draw_triangle(const TriangleType& source_triangle, std::array<grh::vec3, 3>& pts_screen_space, draw_horizontal_line_function<TriangleType> auto&& draw_hline_function, unsigned_integral auto frame_width, unsigned_integral auto frame_height)
		{
			draw_triangle<IteratorCollection, TriangleType>(source_triangle, pts_screen_space, draw_hline_function, frame_width, frame_height, 1u, 1u, (frame_width-1), (frame_height-1));
		}

		constexpr grh::mat4x4 create_perspective(float fovy, float aspect, float z_near, float z_far)
		{
			assert(std::abs(aspect - std::numeric_limits<float>::epsilon()) > 0.0f);

			const float fovy_over_2 = fovy / 2.0f;

#ifndef HAS_GCEM
			const float tanHalfFovy = std::tan(fovy_over_2);
#else

			const float tanHalfFovy = std::is_constant_evaluated() ? gcem::tan(fovy_over_2) : std::tan(fovy_over_2);
#endif

			grh::mat4x4 result = {};
			result[0][0] = 1.0f / (aspect * tanHalfFovy);
			result[1][1] = 1.0f / (tanHalfFovy);
			result[2][2] = (z_far + z_near) / (z_far - z_near);
			result[2][3] = 1.0f;
			result[3][2] = - (2.0f * z_far * z_near) / (z_far - z_near);
			//result[3][3] = 1.0f; maybe ?
			return result;
		}

		template<vector3 Vec3Type>
		constexpr grh::mat4x4 create_lookat(const Vec3Type& eye, const Vec3Type& center, const Vec3Type& up)
		{
			const Vec3Type f(normalize(sub_xyz(center,eye)));
			const Vec3Type s(normalize(cross(up, f)));
			const Vec3Type u(cross(f, s));

			grh::mat4x4 result = {};
			result[0][0] = s.x;
			result[1][0] = s.y;
			result[2][0] = s.z;
			result[0][1] = u.x;
			result[1][1] = u.y;
			result[2][1] = u.z;
			result[0][2] = f.x;
			result[1][2] = f.y;
			result[2][2] = f.z;
			result[3][0] = -dot(s, eye);
			result[3][1] = -dot(u, eye);
			result[3][2] = -dot(f, eye);
			result[3][3] = 1.0f;
			return result;
		}

		template<typename IteratorCollection, triangle_list TriangleList>
		constexpr void render_rasterize(const TriangleList& triangles, const camera auto& camera, draw_horizontal_line_function<triangle_from_list_t<TriangleList>> auto&& draw_hline_function, unsigned_integral auto frame_width, unsigned_integral auto frame_height)
		{
			using TriangleType = std::decay_t<decltype(*triangles.begin())>;

			const float target_width_flt 	= static_cast<float>(frame_width);  // NOLINT
			const float target_height_flt 	= static_cast<float>(frame_height); // NOLINT
			const float aspect = target_width_flt / target_height_flt;

			const float near_plane 	= 0.1f;
			const float far_plane 	= 100.0f;

			const grh::mat4x4 perspective 	= create_perspective(camera.fov, aspect, near_plane / 10.0f, far_plane);
			const grh::mat4x4 lookat 		= create_lookat(glm::vec3{camera.pos.x, camera.pos.y, camera.pos.z}, glm::vec3{camera.lookat.x, camera.lookat.y, camera.lookat.z}, glm::vec3{camera.up.x, camera.up.y, camera.up.z});
			const grh::mat4x4 projview 		= mul(perspective, lookat);

			const vector3 auto near_clipping_plane_normal 	= direction_to(camera.pos, camera.lookat);
			const vector3 auto near_clipping_plane_pos 		= add_xyz(camera.pos, mul_xyz(near_clipping_plane_normal, near_plane));

			for(const triangle auto& tri : triangles)
			{
				std::array<TriangleType, 2> clipped_tris; // NOLINT
				size_t num_clipped_tris = 0;
				clipped_tris[num_clipped_tris++] = tri;

				// clip near
				num_clipped_tris += clip_triangle(near_clipping_plane_pos, near_clipping_plane_normal, clipped_tris[0], clipped_tris[1]);

				for(size_t clipped_tri_index=0; clipped_tri_index < num_clipped_tris; clipped_tri_index++)
				{
					const triangle auto& clipped_tri = clipped_tris[clipped_tri_index];
					const grh::vec4 p0 = {clipped_tri.p0.x, clipped_tri.p0.y, clipped_tri.p0.z, 1};
					const grh::vec4 p1 = {clipped_tri.p1.x, clipped_tri.p1.y, clipped_tri.p1.z, 1};
					const grh::vec4 p2 = {clipped_tri.p2.x, clipped_tri.p2.y, clipped_tri.p2.z, 1};

					const grh::vec4 p0_projview = mul(projview, p0);
					const grh::vec4 p1_projview = mul(projview, p1);
					const grh::vec4 p2_projview = mul(projview, p2);

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

					float cross_z = (pts[1].x - pts[0].x) * (pts[2].y - pts[0].y) - (pts[2].x - pts[0].x) * (pts[1].y - pts[0].y);
					const bool backface_culling = cross_z > 0.0f;

					if(backface_culling)
					{
						draw_triangle<IteratorCollection, TriangleType>(clipped_tri, pts, draw_hline_function, frame_width, frame_height);
					}
				}
			}
		}
	}

	template<typename IteratorCollection, triangle_list TriangleList>
	constexpr void 	render(const TriangleList& triangles, const camera auto& camera, draw_horizontal_line_function<triangle_from_list_t<TriangleList>> auto&& draw_hline_function, unsigned_integral auto frame_width, unsigned_integral auto frame_height)
	{
		detail::render_rasterize(triangles, camera, draw_hline_function, frame_width, frame_height);
	}

	struct dummy_iterator_collection { void for_each(auto){} };

	template<triangle_list TriangleList>
	constexpr void 	render(const TriangleList& triangles, const camera auto& camera, draw_horizontal_line_function<triangle_from_list_t<TriangleList>> auto&& draw_hline_function, unsigned_integral auto frame_width, unsigned_integral auto frame_height)
	{
		detail::render_rasterize<dummy_iterator_collection, TriangleList>(triangles, camera, draw_hline_function, frame_width, frame_height);
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
}



#endif //CPURAYTRACER_GT_HPP
