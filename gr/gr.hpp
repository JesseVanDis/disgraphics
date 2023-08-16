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

		constexpr inline int y_pos_at_intersection(int current_ypos, const line_it& a, const line_it& b)
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
			assert(ctx.px_x_to 		>= ctx.px_x_from);
			assert(ctx.px_y 		< ctx.buffer_height);
			assert(ctx.px_x_from 	< ctx.buffer_width);
			assert(ctx.px_x_to 		< ctx.buffer_width);
#else
			ctx.px_x_to		= std::max(ctx.px_x_to, (ctx.px_x_from));
			ctx.px_y 		= std::min(ctx.px_y, (ctx.buffer_height-1));
			ctx.px_x_from 	= std::min(ctx.px_x_from, (ctx.buffer_width-1));
			ctx.px_x_to 	= std::min(ctx.px_x_to, (ctx.buffer_width-1));
#endif
		}

		/// no bounds checking here!
		template<triangle TriangleType>
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
				line_it line_it_long = line_it_from_line(line_long.p0, line_long.p1);
				line_it line_it_top  = line_it_from_line(line_top.p0, line_top.p1);
				line_it line_it_bot  = line_it_from_line(line_bot.p0, line_bot.p1);

				int y=line_it_long.y_start;
				for(; y<line_it_long.y_start+line_it_top.height; y++)
				{
					ctx.px_y 		= y;
					ctx.px_x_from 	= static_cast<int>(line_it_long.x);
					ctx.px_x_to 	= static_cast<int>(line_it_top.x);

					check_context_validity(ctx);
					draw_hline_function(source_triangle, ctx);

					++line_it_long;
					++line_it_top;
				}

				const int yLimit = y_pos_at_intersection(y, line_it_long, line_it_bot);
				for(; y<yLimit; y++)
				{
					ctx.px_y 		= y;
					ctx.px_x_from 	= static_cast<int>(line_it_long.x);
					ctx.px_x_to 	= static_cast<int>(line_it_bot.x);

					check_context_validity(ctx);
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

					check_context_validity(ctx);
					draw_hline_function(source_triangle, ctx);

					++line_it_long;
					++line_it_top;
				}

				const int yLimit = y_pos_at_intersection(y, line_it_long, line_it_bot);
				for(; y<yLimit; y++)
				{
					ctx.px_y 		= y;
					ctx.px_x_from 	= static_cast<int>(line_it_bot.x);
					ctx.px_x_to 	= static_cast<int>(line_it_long.x);

					check_context_validity(ctx);
					draw_hline_function(source_triangle, ctx);

					++line_it_long;
					++line_it_bot;
				}
			}
		}

#if 0

struct Face
{
	Vec3  p[3];
	Vec3  color = Vec3(0.7f, 0.7f, 0.7f);
	struct
	{
		Vec3 edge0; // a to b
		Vec3 edge1; // b to c
		Vec3 edge2; // c to a
		Vec3 normal;
		Vec3 oneOverNormal;
		Vec3 center;
		float dotNp0 = 0.0;
	} precalculated;

	void precalculate();
};



static uint clip(const Vec3& planePos, const Vec3& planeNormal, const ViewSpaceFace& face, ViewSpaceFace output[2])
{
	Vec3 a = face.p[0];
	Vec3 b = face.p[1];
	Vec3 c = face.p[2];
	Vec3 aToB = face.precalculated.edge0.normalized0(); // TODO: This can be precalculated in the face
	Vec3 bToC = face.precalculated.edge1.normalized0(); // TODO: This can be precalculated in the face
	Vec3 cToA = face.precalculated.edge2.normalized0(); // TODO: This can be precalculated in the face
	float aToBt = intersect(a, aToB, planePos, planeNormal);
	float bToCt = intersect(b, bToC, planePos, planeNormal);
	float cToAt = intersect(c, cToA, planePos, planeNormal);

	bool intersectsAToB = (aToBt > 0.0f && aToBt < face.precalculated.edge0.length());
	bool intersectsBToC = (bToCt > 0.0f && bToCt < face.precalculated.edge1.length());
	bool intersectsCToA = (cToAt > 0.0f && cToAt < face.precalculated.edge2.length());

	Vec3 intersectionAToB = intersectsAToB ? (a + aToB * aToBt) : Vec3::sInvalidPoint();
	Vec3 intersectionBToC = intersectsBToC ? (b + bToC * bToCt) : Vec3::sInvalidPoint();
	Vec3 intersectionCToA = intersectsCToA ? (c + cToA * cToAt) : Vec3::sInvalidPoint();

	const Vec3& triangleTip = (intersectsAToB && intersectsCToA) ? a : ((intersectsAToB && intersectsBToC) ? b : c);
	const Vec3& anyOtherTip = (intersectsAToB && intersectsCToA) ? b : ((intersectsAToB && intersectsBToC) ? c : a);

	if(intersectsAToB || intersectsBToC || intersectsCToA)
	{
		bool shouldCutToQuad = dot((triangleTip - anyOtherTip), planeNormal) < 0.0f;

		const Vec3 points[6] = { a, b, c, intersectionAToB, intersectionBToC, intersectionCToA };
		enum
		{
			IndexA = 0,
			IndexB = 1,
			IndexC = 2,
			IndexIntersectAtoB = 3,
			IndexIntersectBtoC = 4,
			IndexIntersectCtoA = 5,
		};

		const static vector<uint> s_pointIndices[] = {
			{ }, 														// 	0, triangle has not intersection after all ?
			{ IndexC, IndexIntersectCtoA, IndexIntersectBtoC }, 		// 	1, intersection between c->a and c->b, cut out triangle
			{ IndexA, IndexIntersectAtoB, IndexIntersectCtoA }, 		// 	2, intersection between a->b and a->c, cut out triangle
			{ IndexB, IndexIntersectAtoB, IndexIntersectBtoC }, 		// 	3, intersection between b->a and b->c, cut out triangle
			{ }, 														// 	4, triangle has not intersection after all ?
			{ IndexA, IndexB, IndexIntersectBtoC, IndexIntersectCtoA }, // 	5, intersection between c->a and c->b, cut out quad
			{ IndexB, IndexC, IndexIntersectCtoA, IndexIntersectAtoB }, // 	6, intersection between a->b and a->c, cut out quad
			{ IndexC, IndexA, IndexIntersectAtoB, IndexIntersectBtoC }, // 	7, intersection between b->a and b->c, cut out quad
		};

		const uint state = 	((!intersectsAToB ? 0b0001u : 0u) |
							   (!intersectsBToC ? 0b0010u : 0u) |
							   (!intersectsCToA ? 0b0011u : 0u)) +
							  (shouldCutToQuad ? 0b0100u : 0u);

		const vector<uint>& indices = s_pointIndices[state];

		for(size_t i=0; i<indices.size()-2; i++)
		{
			ViewSpaceFace& target = output[i];
			target.p[0] = points[indices[((i<<1u)+0u)&0b11u]];
			target.p[1] = points[indices[((i<<1u)+1u)&0b11u]];
			target.p[2] = points[indices[((i<<1u)+2u)&0b11u]];
			target.precalculate();
		}
		uint numTris = indices.size() - 2;
		return numTris;
	}
	else
	{
		output[0] = face;
		return dot((c - planePos), planeNormal) > 0.0f ? 1 : 0u;
	}
}

static void clip(const Vec3& planePos, const Vec3& planeNormal, ViewSpaceFace buffer[MaxClippingPlanes*2], size_t* pNumPlanes)
{
	ViewSpaceFace output[MaxClippingPlanes*2];
	size_t numPlanes = 0u;
	for(size_t i=0; i < *pNumPlanes; i++)
	{
		numPlanes += clip(planePos, planeNormal, buffer[i], &output[numPlanes]);
		assert(numPlanes <= MaxClippingPlanes*2);
	}
	*pNumPlanes = numPlanes;
	memcpy(buffer, output, sizeof(output[0]) * numPlanes);
}

#endif

		constexpr float intersect(const grh::vec3& ray_origin, const grh::vec3& ray_dir, const grh::vec3& plane_pos, const grh::vec3& plane_normal)
		{
			const float denom = (plane_normal.x * ray_dir.x) + (plane_normal.y * ray_dir.y) + (plane_normal.z * ray_dir.z);
			if ((denom*denom) > (0.0001f * 0.0001f)) // your favorite epsilon
			{
				const grh::vec3 d = {plane_pos.x - ray_origin.x, plane_pos.y - ray_origin.y, plane_pos.z - ray_origin.z};
				const float d_dot = (d.x * plane_normal.x) + (d.y * plane_normal.y) + (d.z * plane_normal.z);
				return d_dot / denom;
			}
			return -1.0f;
		}

		constexpr bool clip_triangle(const grh::vec3& plane_pos, const grh::vec3& plane_normal, std::array<grh::vec3, 3>& tri_in_out, std::array<grh::vec3, 3>& tri_extra_out)
		{
			const grh::vec3& a = tri_in_out[0];
			const grh::vec3& b = tri_in_out[1];
			const grh::vec3& c = tri_in_out[2];

			const grh::vec3 ab = {b.x-a.x, b.y-a.y, b.z-a.z};
			const grh::vec3 bc = {c.x-b.x, c.y-b.y, c.z-b.z};
			const grh::vec3 ca = {a.x-c.x, a.y-c.y, a.z-c.z};

			const float ab_len = std::sqrt(ab.x*ab.x + ab.y*ab.y + ab.z*ab.z);
			const float bc_len = std::sqrt(bc.x*bc.x + bc.y*bc.y + bc.z*bc.z);
			const float ca_len = std::sqrt(ca.x*ca.x + ca.y*ca.y + ca.z*ca.z);

			const float ab_len_inv = 1.0f / ab_len;
			const float bc_len_inv = 1.0f / bc_len;
			const float ca_len_inv = 1.0f / ca_len;

			const grh::vec3 ab_dir = {ab.x * ab_len_inv, ab.y * ab_len_inv, ab.z * ab_len_inv}; // TODO: This can be precalculated
			const grh::vec3 bc_dir = {bc.x * bc_len_inv, bc.y * bc_len_inv, bc.z * bc_len_inv}; // TODO: This can be precalculated
			const grh::vec3 ca_dir = {ca.x * ca_len_inv, ca.y * ca_len_inv, ca.z * ca_len_inv}; // TODO: This can be precalculated

			const float a_to_b_t = intersect(a, ab_dir, plane_pos, plane_normal);
			const float b_to_c_t = intersect(b, bc_dir, plane_pos, plane_normal);
			const float c_to_a_t = intersect(c, ca_dir, plane_pos, plane_normal);

			const bool intersects_a_to_b = (a_to_b_t > 0.0f && a_to_b_t < ab_len);
			const bool intersects_b_to_c = (b_to_c_t > 0.0f && b_to_c_t < bc_len);
			const bool intersects_c_to_a = (c_to_a_t > 0.0f && c_to_a_t < ca_len);

			const grh::vec3 intersection_a_to_b = {a.x + ab_dir.x * a_to_b_t, a.y + ab_dir.y * a_to_b_t, a.z + ab_dir.z * a_to_b_t};
			const grh::vec3 intersection_b_to_c = {b.x + bc_dir.x * b_to_c_t, b.y + bc_dir.y * b_to_c_t, b.z + bc_dir.z * b_to_c_t};
			const grh::vec3 intersection_c_to_a = {c.x + ca_dir.x * c_to_a_t, c.y + ca_dir.y * c_to_a_t, c.z + ca_dir.z * c_to_a_t};

			const grh::vec3& triangle_tip  = (intersects_a_to_b && intersects_c_to_a) ? a : ((intersects_a_to_b && intersects_b_to_c) ? b : c);
			const grh::vec3& any_other_tip = (intersects_a_to_b && intersects_c_to_a) ? b : ((intersects_a_to_b && intersects_b_to_c) ? c : a);

			if(intersects_a_to_b || intersects_b_to_c || intersects_c_to_a)
			{
				const grh::vec3 	tip_to_tip 			= {triangle_tip.x - any_other_tip.x, triangle_tip.y - any_other_tip.y, triangle_tip.z - any_other_tip.z};
				const bool 			should_cut_to_quad 	= (tip_to_tip.x*plane_normal.x + tip_to_tip.y*plane_normal.y + tip_to_tip.z*plane_normal.z) < 0.0f;
				const grh::vec3 	points[6] 			= { a, b, c, intersection_a_to_b, intersection_b_to_c, intersection_c_to_a};

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
				tri_in_out[0] = points[indices.indices[((0<<1u)+0u)&0b11u]];
				tri_in_out[1] = points[indices.indices[((0<<1u)+1u)&0b11u]];
				tri_in_out[2] = points[indices.indices[((0<<1u)+2u)&0b11u]];

				if(num_tris == 2)
				{
					tri_extra_out[0] = points[indices.indices[((1<<1u)+0u)&0b11u]];
					tri_extra_out[1] = points[indices.indices[((1<<1u)+1u)&0b11u]];
					tri_extra_out[2] = points[indices.indices[((1<<1u)+2u)&0b11u]];
					return true;
				}
			}

			return false;
		}

		template<triangle TriangleType>
		constexpr void draw_triangle(const TriangleType& source_triangle, std::array<grh::vec3, 3>& pts_screen_space, draw_horizontal_line_function<TriangleType> auto&& draw_hline_function, unsigned_integral auto frame_width, unsigned_integral auto frame_height,
									 unsigned_integral auto viewport_x_start, unsigned_integral auto viewport_y_start, unsigned_integral auto viewport_x_end, unsigned_integral auto viewport_y_end)
		{
			// clipping on edges

			std::array<std::array<grh::vec3, 3>, (2*2*2*2)+1> 	clipped_tris; // NOLINT
			std::uint_fast8_t 									clipped_tris_num = 0;

			// first just add the main triangle
			clipped_tris[clipped_tris_num++] = pts_screen_space;

			// clipping planes
			//struct clipping_plane { grh::vec3 origin, dir; };
			//clipping_plane clipping

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
				draw_triangle_unsafe(source_triangle, clipped_tris[i], draw_hline_function, frame_width, frame_height);
			}
		}

		template<triangle TriangleType>
		constexpr void draw_triangle(const TriangleType& source_triangle, std::array<grh::vec3, 3>& pts_screen_space, draw_horizontal_line_function<TriangleType> auto&& draw_hline_function, unsigned_integral auto frame_width, unsigned_integral auto frame_height)
		{
			draw_triangle<TriangleType>(source_triangle, pts_screen_space, draw_hline_function, frame_width, frame_height, 50u, 50u, (frame_width-50), (frame_height-50));
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
		constexpr Vec3Type normalize(Vec3Type v)
		{
			const auto l = static_cast<std::decay_t<decltype(v.x)>>(1) / std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
			return {v.x / l, v.y / l, v.z / l};
		}

		template<vector3 Vec3Type>
		constexpr Vec3Type cross(const Vec3Type& a, const Vec3Type& b)
		{
			return {a.b * b.z - b.b * a.z, a.z * b.a - b.z * a.a, a.a * b.b - b.a * a.b};
		}

		template<vector3 Vec3Type>
		constexpr Vec3Type sub(const Vec3Type& a, const Vec3Type& b)
		{
			return {a.x - b.x, a.y - b.y, a.z - b.z};
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

		template<vector3 Vec3Type>
		constexpr grh::mat4x4 create_lookat(const Vec3Type& eye, const Vec3Type& center, const Vec3Type& up)
		{
			const Vec3Type f(normalize(sub(center,eye)));
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

		template<triangle_list TriangleList>
		constexpr void render_rasterize(const TriangleList& triangles, const camera auto& camera, draw_horizontal_line_function<triangle_from_list_t<TriangleList>> auto&& draw_hline_function, unsigned_integral auto frame_width, unsigned_integral auto frame_height)
		{
			const float target_width_flt 	= static_cast<float>(frame_width);  // NOLINT
			const float target_height_flt 	= static_cast<float>(frame_height); // NOLINT
			const float aspect = target_width_flt / target_height_flt;

			const grh::mat4x4 perspective 	= create_perspective(camera.fov, aspect, 0.1f, 100.0f);
			const grh::mat4x4 lookat 		= create_lookat(glm::vec3{camera.pos.x, camera.pos.y, camera.pos.z}, glm::vec3{camera.lookat.x, camera.lookat.y, camera.lookat.z}, glm::vec3{camera.up.x, camera.up.y, camera.up.z});
			const grh::mat4x4 projview 		= mul(perspective, lookat);

			for(const triangle auto& tri : triangles)
			{
				const grh::vec4 p0 = {tri.p0.x, tri.p0.y, tri.p0.z, 1};
				const grh::vec4 p1 = {tri.p1.x, tri.p1.y, tri.p1.z, 1};
				const grh::vec4 p2 = {tri.p2.x, tri.p2.y, tri.p2.z, 1};

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
					if(pts[0].z > 0.0f || pts[1].z > 0.0f || pts[2].z > 0.0f)
					{
						draw_triangle(tri, pts, draw_hline_function, frame_width, frame_height);
					}
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

#if 1
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

		//static_assert(test());

	}
#endif
}



#endif //CPURAYTRACER_GT_HPP
