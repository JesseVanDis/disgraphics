#ifndef CPURAYTRACER_GT_HPP
#define CPURAYTRACER_GT_HPP

#include <array>
#include <cstdint>
#include <memory>
#include <span>

namespace dis
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

	template<size_t Index>
	constexpr auto& get_tri_pt(triangle auto& tri) requires (Index < 3)
	{
		if constexpr(requires{tri[Index];})
		{
			return tri[Index];
		}
		else if	constexpr(Index == 0)	return tri.p0;
		else if constexpr(Index == 1)	return tri.p1;
		else							return tri.p2;
	}

	template<size_t Index>
	constexpr const auto& get_tri_pt(const triangle auto& tri) requires (Index < 3)
	{
		if constexpr(requires{tri[Index];})
		{
			return tri[Index];
		}
		else if	constexpr(Index == 0)	return tri.p0;
		else if constexpr(Index == 1)	return tri.p1;
		else							return tri.p2;
	}

	template<triangle T>
	using vertex_from_tri_t = std::decay_t<decltype(get_tri_pt<0>(std::declval<T>()))>;

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
		{v.line_length_px} 	-> unsigned_integral;
	};
}

namespace dis::helpers
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
namespace dish = dis::helpers;

namespace dis
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
			uint32_t buffer_width;
			uint32_t buffer_height;
			uint32_t px_y, px_x_from, line_length_px;
		};
		static_assert(draw_horizontal_line_ctx<draw_horizontal_line_ctx_example>);
	}

	template<typename FuncT, typename draw_ctx_t, typename triangle_t>
	concept draw_horizontal_line_function = requires(FuncT v)
	{
		{v(std::declval<triangle_t>(), std::declval<draw_ctx_t>())};
	};

	template<draw_horizontal_line_ctx draw_ctx_t, triangle_list triangle_list_t>
	constexpr void 	render(const triangle_list_t& triangles, const camera auto& camera, draw_horizontal_line_function<draw_ctx_t, triangle_from_list_t<triangle_list_t>> auto&& draw_hline_function, unsigned_integral auto frame_width, unsigned_integral auto frame_height);

	template<triangle_list triangle_list_t>
	constexpr void 	render_nonshaded(const triangle_list_t& triangles, const camera auto& camera, draw_horizontal_line_function<draw_hline_ctx, triangle_from_list_t<triangle_list_t>> auto&& draw_hline_function, unsigned_integral auto frame_width, unsigned_integral auto frame_height);
}

namespace dis::helpers
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

namespace dis
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
            const dish::vec3& p0;
            const dish::vec3& p1;
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

		template<typename T>
		concept line_custom_iterable = requires(T v)
		{
			{v.template get<0>()} 		-> std::same_as<float&>;
		};

		template<typename vertex_t, typename T>
		concept can_assign_vertex = requires(T v)
		{
			{v = std::declval<vertex_t>()} -> std::same_as<T&>;
		};

		template<typename draw_ctx_t>
		concept has_user_defined_iterators = requires(draw_ctx_t v)
		{
			{v.begin} -> line_custom_iterable;
		};

		template<vector3 vertex_t>
		struct screen_space_vertex
		{
			dish::vec3 screen_pos;
			vertex_t  vertex;
		};

		template<triangle triangle_t>
		using screen_space_triangle = std::array<screen_space_vertex<vertex_from_tri_t<triangle_t>>, 3>;

		struct line_it_base
		{
			int y_start, height;

			float x_it, x;
			float z_it, z;

			void increment_base()
			{
				x += x_it;
				z += z_it;
			}

			struct set_precalculated
			{
				float y_start_ceiled, height_ceiled, one_over_height_ceiled, sub_pixel;
			};

			template<vector3 vertex_t>
			set_precalculated set_base(const screen_space_vertex<vertex_t>& p0, const screen_space_vertex<vertex_t>& p1)
			{
				set_precalculated c {
					.y_start_ceiled 		= std::ceil(p0.screen_pos.y),
					.height_ceiled 			= std::ceil(p1.screen_pos.y) - c.y_start_ceiled,
					.one_over_height_ceiled = c.height_ceiled != 0.0f ? (1.0f / c.height_ceiled) : 0.0f,
					.sub_pixel 				= c.y_start_ceiled - p0.screen_pos.y
				};

				//assert(height_ceiled != 0.0f); // this is going to be a division over 0 ! // TODO: handle this to avoid NaN

				y_start    = static_cast<int>(c.y_start_ceiled);
				height     = static_cast<int>(c.height_ceiled);
				x_it 		= (p1.screen_pos.x - p0.screen_pos.x) * c.one_over_height_ceiled;
				x			= p0.screen_pos.x + (x_it * c.sub_pixel);
				z_it 		= (p1.screen_pos.z - p0.screen_pos.z) * c.one_over_height_ceiled;
				z			= p0.screen_pos.z + (z_it * c.sub_pixel);

				return c;
			}
		};

		template<unsigned int index>
		inline void add(line_custom_iterable auto& a, line_custom_iterable auto& b)
		{
			if constexpr(requires{a.template get<index>();})
			{
				(a.template get<index>()) += (b.template get<index>());
				add<index+1>(a, b);
			}
		}

		struct UV
		{
			float u,v;
		};

		template<typename user_defined_iterators_t, typename enable = void>
		struct line_it;

		template<typename user_defined_iterators_t>
		struct line_it<user_defined_iterators_t, std::enable_if_t<line_custom_iterable<user_defined_iterators_t>>> : line_it_base
		{
			user_defined_iterators_t user_defined;
			user_defined_iterators_t user_defined_it;

			float one_over_z;

			UV p0_uv;
			UV p1_uv;
			UV p0_uv_over_z;

			/*
			const SVert& StartPoint = TriInfo.TriMesh.v[Line.StartVertIndex];
			CVec2f UVOverZStart = CVec2f(StartPoint.uv.X, StartPoint.uv.Y) * OneOverZStart;

			CEdgeStart EdgeStart;
			const SVert& StartPoint = TriInfo.TriMesh.v[Line.StartVertIndex];
			const SVert& EndPoint = TriInfo.TriMesh.v[Line.EndVertIndex];


			float Z1 = StartPoint.p.Z, Z2 = EndPoint.p.Z;
			float OneOverZStart = 1.0f / Z1, OneOverZEnd = 1.0f / Z2;
			CVec2f UVOverZStart = CVec2f(StartPoint.uv.X, StartPoint.uv.Y) * OneOverZStart;
			CVec2f UVOverZEnd = CVec2f(EndPoint.uv.X, EndPoint.uv.Y) * OneOverZEnd;
			EdgeStart.Recalculate(OneOverZStart, OneOverZEnd, UVOverZStart, UVOverZEnd, Line.StartPoint, Line.EndPoint);

			 UVOverZ = UVOverZStart
			 */


			void increment()
			{
				increment_base();
				add<0>(user_defined, user_defined_it);
			}

			template<vector3 vertex_t>
			void set(const screen_space_vertex<vertex_t>& p0, const screen_space_vertex<vertex_t>& p1)
			{
				const set_precalculated c = set_base(p0, p1);

				one_over_z = 1.0f / p0.screen_pos.z;

				p0_uv.u = p0.vertex.u;
				p0_uv.v = p0.vertex.v;

				p1_uv.u = p1.vertex.u;
				p1_uv.v = p1.vertex.v;


				const float over_over_z_start = 1.0f / p0.screen_pos.z;
				const float over_over_z_end = 1.0f / p1.screen_pos.z;

				p0_uv_over_z.u = p0_uv.u * over_over_z_start;
				p0_uv_over_z.v = p0_uv.v * over_over_z_start;

				// TODO: Set 'user_defined_it'
				// TODO: Set 'user_defined'
				/*
				const user_defined_iterators_t it0 = p0;
				user_defined_it = p1;
				user_defined_it -= it0;
				user_defined_it *= c.one_over_height_ceiled;
				user_defined = user_defined_it;
				user_defined *= c.sub_pixel;
				user_defined += it0;
				 */
			}
		};

			/*
			 inline CEdgeStart GetPrepairedStartingEdge(const SLine& Line, const STriangleInfo& TriInfo)
{
	CEdgeStart EdgeStart;
	const SVert& StartPoint = TriInfo.TriMesh.v[Line.StartVertIndex];
	const SVert& EndPoint = TriInfo.TriMesh.v[Line.EndVertIndex];


	float Z1 = StartPoint.p.Z, Z2 = EndPoint.p.Z;
	float OneOverZStart = 1.0f / Z1, OneOverZEnd = 1.0f / Z2;
	CVec2f UVOverZStart = CVec2f(StartPoint.uv.X, StartPoint.uv.Y) * OneOverZStart;
	CVec2f UVOverZEnd = CVec2f(EndPoint.uv.X, EndPoint.uv.Y) * OneOverZEnd;
	EdgeStart.Recalculate(OneOverZStart, OneOverZEnd, UVOverZStart, UVOverZEnd, Line.StartPoint, Line.EndPoint);
	return EdgeStart;
}


			 	STriangleSection ts;

	ts.Buffer = Screen.mPixelBuffer;
	ts.DepthBuffer = Screen.GetDepthBuffer();
	ts.LeadingEdgeStart = Tri.LeadingEdgeStart;
	ts.TopY = ts.LeadingEdgeStart.YStart;
	ts.EndY = ts.TopY+ts.LeadingEdgeStart.Height;
	ts.Pitch = Screen.mPitch;
	ts.RemainingVertDirection = RemainingVertDirection;
	ts.StartY = ts.TopY;
	ts.TailingLineEdgeStart1 = Tri.TailingLineEdgeStart1;
	ts.TailingLineEdgeStart2 = Tri.TailingLineEdgeStart2;
	ts.TextureSize = sTextureList[sBindedTexture].TextSize;
	ts.TextSizeXMin1 = ts.TextureSize.X-1;
	ts.TextSizeYMin1 = ts.TextureSize.Y-1;
	ts.TextureBuffer = (unsigned char*)sTextureList[sBindedTexture].TextBuffer;
	ts.TextureSizeXf = (float)ts.TextureSize.X;
	ts.TextureSizeYf = (float)ts.TextureSize.Y;
	ts.EndX = (int)Tri.LeadingEdgeStart.EndPoint.X;

	//AddRenderJob(ts);
	ts.Draw();



typedef struct
{
	int TopY;
	int StartY, EndY, EndX;
	float OneOverZEnd;
	CEdgeStart LeadingEdgeStart;
	CEdgeStart TailingLineEdgeStart1;
	CEdgeStart TailingLineEdgeStart2;
	int RemainingVertDirection;

	float* DepthBuffer;
	UInt* Buffer;
	UInt Pitch;

	float TextureSizeXf;
	float TextureSizeYf;
	CVec2i TextureSize;
	int TextSizeXMin1;
	int TextSizeYMin1;
	unsigned char* TextureBuffer;
	unsigned char* OrTextureBuffer;


	inline void Draw()
	{
		unsigned char LocalTexBuffer[128*128];
		OrTextureBuffer = TextureBuffer;
		TextureBuffer = LocalTexBuffer;
		memcpy(TextureBuffer, OrTextureBuffer, TextureSize.X * TextureSize.Y * 3);

		CEdgeStart* RightEdge = NULL;
		CEdgeStart* LeftEdge = NULL;
		if(RemainingVertDirection > 0)
		{
			LeftEdge = TailingLineEdgeStart1.Height <= 0 ? &TailingLineEdgeStart2 : &TailingLineEdgeStart1;
			RightEdge = &LeadingEdgeStart;
		}
		else
		{
			LeftEdge = &LeadingEdgeStart;
			RightEdge = TailingLineEdgeStart1.Height <= 0 ? &TailingLineEdgeStart2 : &TailingLineEdgeStart1;
		}

		__m128 TextureSizeXf4 = _mm_set_ps1(TextureSizeXf);
		__m128 TextureSizeYf4 = _mm_set_ps1(TextureSizeYf);
		__m128i TextureSizeXi4 = _mm_set1_epi32(TextureSize.X);
		__m128i TextureSizeYi4 = _mm_set1_epi32(TextureSize.Y);
		__m128i TextSizeXMinOne4 = _mm_set1_epi32(TextSizeXMin1);
		__m128i TextSizeYMinOne4 = _mm_set1_epi32(TextSizeYMin1);

		int LeadingEdgeStartY = LeadingEdgeStart.YStart;

		//if((*DepthBufferPos) < Z)

		//float WidthLeading = ((float)EndX - LeadingEdgeStart.X);

		//UInt BufferPosVert1 = LeadingEdgeStart.X + (LeadingEdgeStart.YStart *Pitch);
		//UInt BufferPosVert2 = TailingLineEdgeStart2.X + (TailingLineEdgeStart2.YStart *Pitch);
		//UInt BufferPosVert3 = EndX + (EndY *Pitch);
		//float Vert1Z = 1.0f / LeadingEdgeStart.OneOverZ;
		//float Vert2Z = 1.0f / TailingLineEdgeStart2.OneOverZ;
		//float Vert3Z = 1.0f / TailingLineEdgeStart2.OneOverZ * TailingLineEdgeStart2.OneOverZIt * TailingLineEdgeStart2.Height;  //OneOverZEnd
		//float* DepthBufferPosVert1 = &DepthBuffer[BufferPosVert1];
		//float* DepthBufferPosVert2 = &DepthBuffer[BufferPosVert2];
		//float* DepthBufferPosVert3 = &DepthBuffer[BufferPosVert3];


		//if(((*DepthBufferPosVert1) > Vert1Z) && ((*DepthBufferPosVert2) > Vert2Z) && ((*DepthBufferPosVert3) > Vert3Z))
		//return;

		for(int Y=StartY; Y<EndY; ++Y)
		{
			int XLeft = (int)(LeftEdge->X);
			int XRight = (int)(RightEdge->X);

			float Width = RightEdge->X - LeftEdge->X;
			float OneOverWidth = 1.0f / Width;

			float	SubTexel = ((float)XLeft) - LeftEdge->X;

			float OneOverZIt = (RightEdge->OneOverZ - LeftEdge->OneOverZ) * OneOverWidth;
			float OneOverZItMul4 = OneOverZIt * 4.0f;
			float OneOverZ = LeftEdge->OneOverZ + (OneOverZIt * SubTexel);

			CVec2f UVOverZIt = (RightEdge->UVOverZ - LeftEdge->UVOverZ) * OneOverWidth;
			CVec2f UVOverZItMul4 = (RightEdge->UVOverZ - LeftEdge->UVOverZ) * OneOverWidth * 4.0f;
			CVec2f UVOverZ = LeftEdge->UVOverZ + (UVOverZIt * SubTexel);

			// get buffer items
			UInt BufferPos = XLeft + (Y*Pitch);
			float* DepthBufferPos = &DepthBuffer[BufferPos];
			UInt* CurrentBufferPos = &Buffer[BufferPos];



			UInt LineWidth = XRight - XLeft;
			UInt RAligned = (XLeft + LineWidth) - (LineWidth & 3);

			//for(UInt X4 = XLeft; X4 < RAligned; X4+=4)
			//{
			//	// defaults
			//	__m128 One4 = _mm_set_ps1(1.0f);
			//	__m128i Three4 = _mm_set1_epi32(3);
			//	__m128 OneOverZ4 = _mm_set_ps(OneOverZ + (OneOverZIt*3.0f), OneOverZ + (OneOverZIt*2.0f), OneOverZ + (OneOverZIt*1.0f), OneOverZ + (OneOverZIt*0.0f));
			//	__m128 UVOverZ4x = _mm_set_ps(UVOverZ.X + (UVOverZIt.X*3.0f), UVOverZ.X + (UVOverZIt.X*2.0f), UVOverZ.X + (UVOverZIt.X*1.0f), UVOverZ.X + (UVOverZIt.X*0.0f));
			//	__m128 UVOverZ4y = _mm_set_ps(UVOverZ.Y + (UVOverZIt.Y*3.0f), UVOverZ.Y + (UVOverZIt.Y*2.0f), UVOverZ.Y + (UVOverZIt.Y*1.0f), UVOverZ.Y + (UVOverZIt.Y*0.0f));

			//	// calc. z
			//	union{__m128 Z4; float Za[4];}; Z4 = _mm_div_ps(One4, OneOverZ4);

			//	// calc. uv
			//	union{__m128 u4; float Ua[4];}; u4 = _mm_mul_ps(Z4, UVOverZ4x);
			//	union{__m128 v4; float Va[4];}; v4 = _mm_mul_ps(Z4, UVOverZ4y);

			//	float* StartDept = DepthBufferPos;
			//	union{__m128 d4; float Da[4];}; d4 = _mm_set_ps((*(DepthBufferPos+3)), (*(DepthBufferPos+2)), (*(DepthBufferPos+1)), (*(DepthBufferPos+0)));

			//	// z read / write
			//	union{__m128 OnTop4; float OnTopa[4]; UInt OnTopia[4];}; OnTop4 = _mm_cmplt_ps(d4, Z4);
			//	__m128 z4And = _mm_and_ps(Z4, OnTop4);
			//	union{__m128i m4i; __m128 m4f;}; m4i = _mm_set1_epi32(0xffffffff);
			//	__m128 t = _mm_xor_ps(m4f, OnTop4);

			//	d4 = _mm_and_ps(d4, t);
			//	d4 = _mm_add_ps(d4, z4And);

			//	// uv to texture point
			//	union{__m128i tu4; UInt tua[4];};
			//	union{__m128i tv4; UInt tva[4];};

			//	//tu4 = _mm_and_si128(_mm_cvtps_epi32(_mm_mul_ps(u4, TextureSizeXf4)), TextSizeXMinOne4);
			//	//tv4 = _mm_and_si128(_mm_cvtps_epi32(_mm_mul_ps(v4, TextureSizeYf4)), TextSizeYMinOne4);

			//	for(UInt i=0; i<4; i++)
			//	{
			//		float u = Ua[i];
			//		float v = Va[i];

			//		UInt tu = (UInt)(u * TextureSizeXf);
			//		UInt tv = (UInt)(v * TextureSizeYf);
			//		tu = tu & TextSizeXMin1;
			//		tv = tv & TextSizeYMin1;

			//		tua[i] = tu;
			//		tva[i] = tv;
			//	}

			//	// color from texture point
			//	union{__m128i Addr4; UInt Addra[4];};
			//	Addr4 = MulInts4(_mm_add_epi32(tu4, MulInts4(tv4, TextureSizeXi4)), Three4);
			//	__m128i c4 = _mm_set_epi32(*((UInt*)&TextureBuffer[Addra[3]]), *((UInt*)&TextureBuffer[Addra[2]]), *((UInt*)&TextureBuffer[Addra[1]]), *((UInt*)&TextureBuffer[Addra[0]]));

			//	union{__m128i r4; UInt ra[4];};
			//	r4 = _mm_and_si128(c4, _mm_set1_epi32(0xffffff));

			//	// write
			//	memcpy(DepthBufferPos, &Da[0], sizeof(float) * 4);

			//	if(OnTopia[0]) (*CurrentBufferPos) = ra[0];
			//	if(OnTopia[1]) (*(CurrentBufferPos+1)) = ra[1];
			//	if(OnTopia[2]) (*(CurrentBufferPos+2)) = ra[2];
			//	if(OnTopia[3]) (*(CurrentBufferPos+3)) = ra[3];

			//	OneOverZ += OneOverZItMul4;
			//	UVOverZ.X += UVOverZItMul4.X;
			//	UVOverZ.Y += UVOverZItMul4.Y;
			//	CurrentBufferPos+=4;
			//	DepthBufferPos+=4;
			//}

			//(*CurrentBufferPos) = 0xffffff;

			//for(int X = RAligned; X <= XRight; ++X)
			for(int X = XLeft; X <= XRight; ++X)
			{
				float Z = (1.0f/OneOverZ);

				if((*DepthBufferPos) < Z)
				{
					float u = UVOverZ.X * Z;
					float v = UVOverZ.Y * Z;
					(*DepthBufferPos) = Z;


					UInt tu = (UInt)(u * TextureSizeXf);
					UInt tv = (UInt)(v * TextureSizeYf);
					tu = tu & TextSizeXMin1;
					tv = tv & TextSizeYMin1;

					unsigned char* ptr = &(TextureBuffer[((tu + (tv*TextureSize.X)) * 3)]);
					UInt c = *(UInt*)ptr;
					c = c & 0xffffff;

					(*CurrentBufferPos) = c;

				}
				OneOverZ += OneOverZIt;
				UVOverZ.X += UVOverZIt.X;
				UVOverZ.Y += UVOverZIt.Y;
				CurrentBufferPos++;
				DepthBufferPos++;
			}

			// interpolate
			RightEdge->X += RightEdge->XIt;
			LeftEdge->X += LeftEdge->XIt;
			RightEdge->OneOverZ += RightEdge->OneOverZIt;
			LeftEdge->OneOverZ += LeftEdge->OneOverZIt;
			RightEdge->UVOverZ.X += RightEdge->UVOverZIt.X;
			RightEdge->UVOverZ.Y += RightEdge->UVOverZIt.Y;
			LeftEdge->UVOverZ.X += LeftEdge->UVOverZIt.X;
			LeftEdge->UVOverZ.Y += LeftEdge->UVOverZIt.Y;

			// swap loading edges if needed
			if((Y-LeadingEdgeStartY) >= RightEdge->Height-1)
				RightEdge = &TailingLineEdgeStart2;
			if((Y-LeadingEdgeStartY) >= LeftEdge->Height-1)
				LeftEdge = &TailingLineEdgeStart2;
		}

		TextureBuffer = OrTextureBuffer;
	}
}STriangleSection;

			 */


		template<typename user_defined_iterators_t>
        struct line_it<user_defined_iterators_t, std::enable_if_t<!line_custom_iterable<user_defined_iterators_t>>> : line_it_base
        {
			void increment()
			{
				increment_base();
			}

			template<vector3 vertex_t>
			void set(const screen_space_vertex<vertex_t>& p0, const screen_space_vertex<vertex_t>& p1)
			{
				set_base(p0, p1);
			}
		};

		template<typename VecA, typename VecB>
		using smallest_vec3_t = std::conditional_t<sizeof(VecA) < sizeof(VecB), VecA, VecB>;

		template<draw_horizontal_line_ctx draw_ctx_t>
		inline constexpr void check_context_validity(draw_ctx_t& ctx)
		{
#if 1
			int to = ctx.px_x_from + ctx.line_length_px;
			assert(to 				>= ctx.px_x_from);
			assert(ctx.px_y 		< ctx.buffer_height);
			assert(ctx.px_x_from 	< ctx.buffer_width);
			assert(to				< ctx.buffer_width);
#else
			ctx.px_y 			= std::min(ctx.px_y, (ctx.buffer_height-1));
			ctx.px_x_from 		= std::min(ctx.px_x_from, (ctx.buffer_width-1));
			ctx.line_length_px 	= (ctx.px_x_from + ctx.line_length_px) > (ctx.buffer_width-1) ? ((ctx.buffer_width-1) - ctx.px_x_from) : ctx.line_length_px;
#endif
		}

		template<draw_horizontal_line_ctx draw_ctx_t, typename user_defined_iterators_t, triangle triangle_t>
		constexpr inline void it_line(const triangle_t& source_triangle, draw_ctx_t& ctx, int y, line_it<user_defined_iterators_t>& left, line_it<user_defined_iterators_t>& right, draw_horizontal_line_function<draw_ctx_t, triangle_t> auto&& draw_hline_function)
		{
			ctx.px_y 			= y;
			ctx.px_x_from 		= static_cast<int>(left.x);
			ctx.line_length_px 	= static_cast<int>(right.x) - ctx.px_x_from;

			check_context_validity(ctx);

			if constexpr(has_user_defined_iterators<draw_ctx_t>)
			{
				const float width = right.x - left.x;
				const float one_over_width = 1.0f / width;
				const float sub_texel = std::floor(left.x) - left.x;

				const float one_over_z_it = (right.one_over_z - left.one_over_z) * one_over_width;
				const float one_over_z 	= left.one_over_z + (one_over_z_it * sub_texel);

				const UV uv_over_z_it 	= {(right.p0_uv_over_z.u - left.p0_uv_over_z.u) * one_over_width, (right.p0_uv_over_z.v - left.p0_uv_over_z.v) * one_over_width};
				const UV uv_over_z 		= {left.p0_uv_over_z.u + (uv_over_z_it.u * sub_texel), left.p0_uv_over_z.v + (uv_over_z_it.v * sub_texel)};

				const float z = (1.0f/one_over_z);

				const float u = uv_over_z.u * z;
				const float v = uv_over_z.v * z;

				if constexpr(requires{ctx.one_over_z;})
				{
					ctx.one_over_z 		= one_over_z;
					ctx.one_over_z_it 	= one_over_z_it;
				}

				ctx.begin.u = u;
				ctx.begin.v = v;
				ctx.it.u = uv_over_z_it.u;
				ctx.it.v = uv_over_z_it.v;
			}

			draw_hline_function(source_triangle, ctx);

			left.increment();
			right.increment();
		}

		template<typename user_defined_iterators_t>
		constexpr inline int y_pos_at_intersection(int current_ypos, const line_it<user_defined_iterators_t>& a, const line_it<user_defined_iterators_t>& b)
		{
			if(b.x_it - a.x_it == 0.0f)
			{
				return (current_ypos-1);
			}
			//assert((b.x_it - a.x_it) != 0.0f);
			int lowest_y 		= std::min(a.y_start + a.height, b.y_start + b.height);
			int iterations_left = static_cast<int>(std::floor((a.x - b.x) / (b.x_it - a.x_it)));
			int y_limit 		= current_ypos + iterations_left;
			return std::min(y_limit, lowest_y);
		}

		/// no bounds checking here!
		template<draw_horizontal_line_ctx draw_ctx_t, triangle triangle_t>
		constexpr void draw_triangle_unsafe(const triangle_t& source_triangle, screen_space_triangle<triangle_t>& triangle, draw_horizontal_line_function<draw_ctx_t, triangle_t> auto&& draw_hline_function, unsigned_integral auto frame_width, unsigned_integral auto frame_height)
		{
			using vertex_t = vertex_from_tri_t<triangle_t>;
			using screen_space_vertex_t = screen_space_vertex<vertex_t>;

			using user_defined_iterators_t = std::conditional_t<has_user_defined_iterators<draw_ctx_t>, std::decay_t<decltype(std::declval<draw_ctx_t>().begin)>, std::nullptr_t>;
			if constexpr(requires{std::declval<draw_ctx_t>().begin;})
			{
				static_assert(has_user_defined_iterators<draw_ctx_t>, "'begin' member found in 'draw_ctx_t' but does not satisfy the 'has_user_defined_iterators' conditions");
			}

            struct line {const screen_space_vertex_t& p0, &p1;};

            std::sort(triangle.begin(), triangle.end(), [](const screen_space_vertex_t& a, const screen_space_vertex_t& b){return a.screen_pos.y < b.screen_pos.y;});

            // take lines
            const line line_long  = {triangle[0], triangle[2]};
            const line line_top   = {triangle[0], triangle[1]};
            const line line_bot   = {triangle[1], triangle[2]};

            // check whether the long line is on the left or right
            float cross_z = (triangle[1].screen_pos.x - triangle[0].screen_pos.x) * (triangle[2].screen_pos.y - triangle[0].screen_pos.y) - (triangle[2].screen_pos.x - triangle[0].screen_pos.x) * (triangle[1].screen_pos.y - triangle[0].screen_pos.y);

			draw_ctx_t ctx; // NOLINT
			ctx.buffer_width = static_cast<uint32_t>(frame_width);
			ctx.buffer_height = static_cast<uint32_t>(frame_height);

			if(cross_z > 0.0f)
			{
				line_it<user_defined_iterators_t> line_it_long, line_it_top, line_it_bot;

				line_it_long.set(line_long.p0, line_long.p1);
				line_it_top.set(line_top.p0, line_top.p1);
				line_it_bot.set(line_bot.p0, line_bot.p1);

				int y=line_it_long.y_start;
				for(; y<line_it_long.y_start+line_it_top.height; y++)
				{
					it_line(source_triangle, ctx, y, line_it_long, line_it_top, draw_hline_function);
				}

				const int yLimit = y_pos_at_intersection(y, line_it_long, line_it_bot);
				for(; y<yLimit; y++)
				{
					it_line(source_triangle, ctx, y, line_it_long, line_it_bot, draw_hline_function);
				}
			}
			else
			{
				line_it<user_defined_iterators_t> line_it_long, line_it_top, line_it_bot;

				line_it_long.set(line_long.p0, line_long.p1);
				line_it_top.set(line_top.p0, line_top.p1);
				line_it_bot.set(line_bot.p0, line_bot.p1);

				int y=line_it_long.y_start;
				for(; y<line_it_long.y_start+line_it_top.height; y++)
				{
					it_line(source_triangle, ctx, y, line_it_top, line_it_long, draw_hline_function);
				}

				const int yLimit = y_pos_at_intersection(y, line_it_long, line_it_bot);
				for(; y<yLimit; y++)
				{
					it_line(source_triangle, ctx, y, line_it_bot, line_it_long, draw_hline_function);
				}
			}
		}

		/// does not check for 0 division
		template<vector3 vector3_t>
		constexpr vector3_t normalize(const vector3_t& v)
		{
			const auto l = static_cast<std::decay_t<decltype(v.x)>>(1) / std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
			return {v.x / l, v.y / l, v.z / l};
		}

		/// does not check for 0 division
		template<vector3 vector3_t>
		constexpr vector3_t direction_to(const vector3_t& from, const vector3_t& to)
		{
			return normalize(vector3_t{to.x - from.x, to.y - from.y, to.z - from.z});
		}

		/// does not check for 0 division
		constexpr auto length(const vector3 auto& vec)
		{
			return std::sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
		}

		constexpr dish::vec3 cross_xyz(const vector3 auto& a, const vector3 auto& b)
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

		constexpr dish::mat4x4 mul(const dish::mat4x4& m1, const dish::mat4x4& m2)
		{
			dish::mat4x4 result;
			for(size_t k=0; k<4; k++) for(size_t i=0; i<4; i++)
				{
					result[k][i] = m1[0][i] * m2[k][0] + m1[1][i] * m2[k][1] + m1[2][i] * m2[k][2] + m1[3][i] * m2[k][3];
				}
			return result;
		}

		constexpr dish::vec4 mul(const dish::mat4x4& m1, const dish::vec4& m2)
		{
			dish::vec4 result = {};
			result.x = m1[0][0] * m2.x + m1[1][0] * m2.y + m1[2][0] * m2.z + m1[3][0] * m2.w;
			result.y = m1[0][1] * m2.x + m1[1][1] * m2.y + m1[2][1] * m2.z + m1[3][1] * m2.w;
			result.z = m1[0][2] * m2.x + m1[1][2] * m2.y + m1[2][2] * m2.z + m1[3][2] * m2.w;
			result.w = m1[0][3] * m2.x + m1[1][3] * m2.y + m1[2][3] * m2.z + m1[3][3] * m2.w;
			return result;
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

		//using vertex_t = vertex_from_tri_t<triangle_t>;
		//using screen_space_vertex_t = screen_space_vertex<vertex_t>;
		//constexpr auto tris_capacity = (2*2*2*2)+1;

		//std::array<std::array<screen_space_vertex_t, 3>, tris_capacity>		clipped_tris; // NOLINT

		template<triangle triangle_t>
		constexpr int clip_triangle(const vector3 auto& plane_pos, const vector3 auto& plane_normal, screen_space_triangle<triangle_t>& tri_in_out, screen_space_triangle<triangle_t>& tri_extra_out)
		{
			const vector3 auto& a = tri_in_out[0].screen_pos;
			const vector3 auto& b = tri_in_out[1].screen_pos;
			const vector3 auto& c = tri_in_out[2].screen_pos;

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

				const unsigned int state = ((!intersects_a_to_b ? 0b0001u : 0u) | (!intersects_b_to_c ? 0b0010u : 0u) | (!intersects_c_to_a ? 0b0011u : 0u)) + (should_cut_to_quad ? 0b0100u : 0u);
				const auto& indices = s_point_indices[state];

				unsigned int num_tris = indices.num_indices - 2;
				assert(num_tris == 1 || num_tris == 2);
				tri_in_out[0].screen_pos = points[indices.indices[((0<<1u)+0u)&0b11u]];
				tri_in_out[1].screen_pos = points[indices.indices[((0<<1u)+1u)&0b11u]];
				tri_in_out[2].screen_pos = points[indices.indices[((0<<1u)+2u)&0b11u]];

				if(num_tris == 2)
				{
					tri_extra_out[0].screen_pos = points[indices.indices[((1<<1u)+0u)&0b11u]];
					tri_extra_out[1].screen_pos = points[indices.indices[((1<<1u)+1u)&0b11u]];
					tri_extra_out[2].screen_pos = points[indices.indices[((1<<1u)+2u)&0b11u]];
					return 1;
				}
			}
			else
			{
				return dot_xyz(sub_xyz(c, plane_pos), plane_normal) > 0 ? 0 : -1;
			}
			return 0;
		}

		template<draw_horizontal_line_ctx draw_ctx_t, triangle triangle_t>
		constexpr void draw_triangle(const triangle_t& source_triangle, const screen_space_triangle<triangle_t>& triangle, draw_horizontal_line_function<draw_ctx_t, triangle_t> auto&& draw_hline_function, unsigned_integral auto frame_width, unsigned_integral auto frame_height,
									 unsigned_integral auto viewport_x_start, unsigned_integral auto viewport_y_start, unsigned_integral auto viewport_x_end, unsigned_integral auto viewport_y_end)
		{
			// clipping on edges

			using vertex_t = vertex_from_tri_t<triangle_t>;
			using screen_space_vertex_t = screen_space_vertex<vertex_t>;
			constexpr auto tris_capacity = (2*2*2*2)+1;

			std::array<std::array<screen_space_vertex_t, 3>, tris_capacity>		clipped_tris; // NOLINT
			std::uint_fast8_t 													clipped_tris_num = 0;

			// first just add the main triangle
			clipped_tris[clipped_tris_num++] = triangle;

			// clipping planes
			// top clipping plane
			{
				const dish::vec3 n = {0, -1, 0};
				const dish::vec3 o = {0, static_cast<float>(viewport_y_end), 0};

				for(std::uint_fast8_t i=clipped_tris_num; i--;)
				{
					std::array<screen_space_vertex_t, 3>& tri = clipped_tris[i];

					const bool 					in_screen[3] 		= {tri[0].screen_pos.y < viewport_y_end, tri[1].screen_pos.y < viewport_y_end, tri[2].screen_pos.y < viewport_y_end};
					const std::uint_fast8_t 	num_pts_in_screen 	= in_screen[0] + in_screen[1] + in_screen[2];

					if(num_pts_in_screen == 0) // not in screen. delete
					{
						std::swap(clipped_tris[i], clipped_tris[clipped_tris_num-1]);
						clipped_tris_num--;
					}
					else if(num_pts_in_screen != 3) // otherwise nothing to clip. leave it
					{
						clipped_tris_num += clip_triangle<triangle_t>(o, n, tri, clipped_tris[clipped_tris_num]);
					}
				}
			}

			// bot clipping plane
			{
				const dish::vec3 n = {0, 1, 0};
				const dish::vec3 o = {0, static_cast<float>(viewport_y_start), 0};

				for(std::uint_fast8_t i=clipped_tris_num; i--;)
				{
					std::array<screen_space_vertex_t, 3>& tri = clipped_tris[i];

					const bool 					in_screen[3] 		= {tri[0].screen_pos.y > viewport_y_start, tri[1].screen_pos.y > viewport_y_start, tri[2].screen_pos.y > viewport_y_start};
					const std::uint_fast8_t 	num_pts_in_screen 	= in_screen[0] + in_screen[1] + in_screen[2];

					if(num_pts_in_screen == 0) // not in screen. delete
					{
						std::swap(clipped_tris[i], clipped_tris[clipped_tris_num-1]);
						clipped_tris_num--;
					}
					else if(num_pts_in_screen != 3) // otherwise nothing to clip. leave it
					{
						clipped_tris_num += clip_triangle<triangle_t>(o, n, tri, clipped_tris[clipped_tris_num]);
					}
				}
			}

			// left clipping plane
			{
				const dish::vec3 n = {1, 0, 0};
				const dish::vec3 o = {static_cast<float>(viewport_x_start), 0, 0};

				for(std::uint_fast8_t i=clipped_tris_num; i--;)
				{
					std::array<screen_space_vertex_t, 3>& tri = clipped_tris[i];

					const bool 					in_screen[3] 		= {tri[0].screen_pos.x > viewport_x_start, tri[1].screen_pos.x > viewport_x_start, tri[2].screen_pos.x > viewport_x_start};
					const std::uint_fast8_t 	num_pts_in_screen 	= in_screen[0] + in_screen[1] + in_screen[2];

					if(num_pts_in_screen == 0) // not in screen. delete
					{
						std::swap(clipped_tris[i], clipped_tris[clipped_tris_num-1]);
						clipped_tris_num--;
					}
					else if(num_pts_in_screen != 3) // otherwise nothing to clip. leave it
					{
						clipped_tris_num += clip_triangle<triangle_t>(o, n, tri, clipped_tris[clipped_tris_num]);
					}
				}
			}

			// right clipping plane
			{
				const dish::vec3 n = {-1, 0, 0};
				const dish::vec3 o = {static_cast<float>(viewport_x_end), 0, 0};

				for(std::uint_fast8_t i=clipped_tris_num; i--;)
				{
					std::array<screen_space_vertex_t, 3>& tri = clipped_tris[i];

					const bool 					in_screen[3] 		= {tri[0].screen_pos.x < viewport_x_end, tri[1].screen_pos.x < viewport_x_end, tri[2].screen_pos.x < viewport_x_end};
					const std::uint_fast8_t 	num_pts_in_screen 	= in_screen[0] + in_screen[1] + in_screen[2];

					if(num_pts_in_screen == 0) // not in screen. delete
					{
						std::swap(clipped_tris[i], clipped_tris[clipped_tris_num-1]);
						clipped_tris_num--;
					}
					else if(num_pts_in_screen != 3) // otherwise nothing to clip. leave it
					{
						clipped_tris_num += clip_triangle<triangle_t>(o, n, tri, clipped_tris[clipped_tris_num]);
					}
				}
			}

			// draw
			for(std::uint_fast8_t i=0; i<clipped_tris_num; i++)
			{
				draw_triangle_unsafe<draw_ctx_t>(source_triangle, clipped_tris[i], draw_hline_function, frame_width, frame_height);
			}
		}

		template<draw_horizontal_line_ctx draw_ctx_t, triangle triangle_t>
		constexpr void draw_triangle(const triangle_t& source_triangle, const screen_space_triangle<triangle_t>& triangle, draw_horizontal_line_function<draw_ctx_t, triangle_t> auto&& draw_hline_function, unsigned_integral auto frame_width, unsigned_integral auto frame_height)
		{
			draw_triangle<draw_ctx_t, triangle_t>(source_triangle, triangle, draw_hline_function, frame_width, frame_height, 1u, 1u, (frame_width-1), (frame_height-1));
		}

		constexpr dish::mat4x4 create_perspective(float fovy, float aspect, float z_near, float z_far)
		{
			assert(std::abs(aspect - std::numeric_limits<float>::epsilon()) > 0.0f);

			const float fovy_over_2 = fovy / 2.0f;

#ifndef HAS_GCEM
			const float tanHalfFovy = std::tan(fovy_over_2);
#else

			const float tanHalfFovy = std::is_constant_evaluated() ? gcem::tan(fovy_over_2) : std::tan(fovy_over_2);
#endif

			dish::mat4x4 result = {};
			result[0][0] = 1.0f / (aspect * tanHalfFovy);
			result[1][1] = 1.0f / (tanHalfFovy);
			result[2][2] = (z_far + z_near) / (z_far - z_near);
			result[2][3] = 1.0f;
			result[3][2] = - (2.0f * z_far * z_near) / (z_far - z_near);
			//result[3][3] = 1.0f; maybe ?
			return result;
		}

		template<vector3 vector3_t>
		constexpr dish::mat4x4 create_lookat(const vector3_t& eye, const vector3_t& center, const vector3_t& up)
		{
			const vector3_t f(normalize(sub_xyz(center,eye)));
			const vector3_t s(normalize(cross(up, f)));
			const vector3_t u(cross(f, s));

			dish::mat4x4 result = {};
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

		template<draw_horizontal_line_ctx draw_ctx_t, triangle_list triangle_list_t>
		constexpr void render_rasterize(const triangle_list_t& triangles, const camera auto& camera, draw_horizontal_line_function<draw_ctx_t, triangle_from_list_t<triangle_list_t>> auto&& draw_hline_function, unsigned_integral auto frame_width, unsigned_integral auto frame_height)
		{
			using triangle_t 			= std::decay_t<decltype(*triangles.begin())>;

			const float target_width_flt 	= static_cast<float>(frame_width);  // NOLINT
			const float target_height_flt 	= static_cast<float>(frame_height); // NOLINT
			const float aspect = target_width_flt / target_height_flt;

			const float near_plane 	= 0.1f;
			const float far_plane 	= 100.0f;

			const dish::mat4x4 perspective 	= create_perspective(camera.fov, aspect, near_plane / 10.0f, far_plane);
			const dish::mat4x4 lookat 		= create_lookat(glm::vec3{camera.pos.x, camera.pos.y, camera.pos.z}, glm::vec3{camera.lookat.x, camera.lookat.y, camera.lookat.z}, glm::vec3{camera.up.x, camera.up.y, camera.up.z});
			const dish::mat4x4 projview 		= mul(perspective, lookat);

			const vector3 auto near_clipping_plane_normal 	= direction_to(camera.pos, camera.lookat);
			const vector3 auto near_clipping_plane_pos 		= add_xyz(camera.pos, mul_xyz(near_clipping_plane_normal, near_plane));

			for(const triangle auto& tri : triangles)
			{
				std::array<screen_space_triangle<triangle_t>, 2> clipped_tris; // NOLINT
				size_t num_clipped_tris = 0;

				clipped_tris[0][0].vertex = get_tri_pt<0>(tri);
				clipped_tris[0][1].vertex = get_tri_pt<1>(tri);
				clipped_tris[0][2].vertex = get_tri_pt<2>(tri);
				clipped_tris[0][0].screen_pos = {clipped_tris[0][0].vertex.x, clipped_tris[0][0].vertex.y, clipped_tris[0][0].vertex.z};
				clipped_tris[0][1].screen_pos = {clipped_tris[0][1].vertex.x, clipped_tris[0][1].vertex.y, clipped_tris[0][1].vertex.z};
				clipped_tris[0][2].screen_pos = {clipped_tris[0][2].vertex.x, clipped_tris[0][2].vertex.y, clipped_tris[0][2].vertex.z};
				num_clipped_tris++;

				// clip near
				num_clipped_tris += clip_triangle<triangle_t>(near_clipping_plane_pos, near_clipping_plane_normal, clipped_tris[0], clipped_tris[1]);

				for(size_t clipped_tri_index=0; clipped_tri_index < num_clipped_tris; clipped_tri_index++)
				{
					screen_space_triangle<triangle_t>& clipped_tris_result = clipped_tris[clipped_tri_index];

					// officially the below should be done, but clip is not ready for this yet ( it only clips on 'screen_pos' level )
#if 0
					get_tri_pt<0>(clipped_tri) = clipped_tris_result[0].vertex;
					get_tri_pt<1>(clipped_tri) = clipped_tris_result[1].vertex;
					get_tri_pt<2>(clipped_tri) = clipped_tris_result[2].vertex;
#else
					triangle_t clipped_tri = tri;
					get_tri_pt<0>(clipped_tri).x = clipped_tris_result[0].screen_pos.x;
					get_tri_pt<0>(clipped_tri).y = clipped_tris_result[0].screen_pos.y;
					get_tri_pt<0>(clipped_tri).z = clipped_tris_result[0].screen_pos.z;
					get_tri_pt<1>(clipped_tri).x = clipped_tris_result[1].screen_pos.x;
					get_tri_pt<1>(clipped_tri).y = clipped_tris_result[1].screen_pos.y;
					get_tri_pt<1>(clipped_tri).z = clipped_tris_result[1].screen_pos.z;
					get_tri_pt<2>(clipped_tri).x = clipped_tris_result[2].screen_pos.x;
					get_tri_pt<2>(clipped_tri).y = clipped_tris_result[2].screen_pos.y;
					get_tri_pt<2>(clipped_tri).z = clipped_tris_result[2].screen_pos.z;
#endif

					const dish::vec4 p0 = {get_tri_pt<0>(clipped_tri).x, get_tri_pt<0>(clipped_tri).y, get_tri_pt<0>(clipped_tri).z, 1};
					const dish::vec4 p1 = {get_tri_pt<1>(clipped_tri).x, get_tri_pt<1>(clipped_tri).y, get_tri_pt<1>(clipped_tri).z, 1};
					const dish::vec4 p2 = {get_tri_pt<2>(clipped_tri).x, get_tri_pt<2>(clipped_tri).y, get_tri_pt<2>(clipped_tri).z, 1};

					const dish::vec4 p0_projview = mul(projview, p0);
					const dish::vec4 p1_projview = mul(projview, p1);
					const dish::vec4 p2_projview = mul(projview, p2);

					assert(p0_projview.w != 0.0f);

					clipped_tris_result[0].screen_pos.x = ((p0_projview.x / p0_projview.w) * 0.5f + 0.5f) * target_width_flt;
					clipped_tris_result[0].screen_pos.y = ((p0_projview.y / p0_projview.w) * 0.5f + 0.5f) * target_height_flt;
					clipped_tris_result[0].screen_pos.z = (p0_projview.z / p0_projview.w);

					clipped_tris_result[1].screen_pos.x = ((p1_projview.x / p1_projview.w) * 0.5f + 0.5f) * target_width_flt;
					clipped_tris_result[1].screen_pos.y = ((p1_projview.y / p1_projview.w) * 0.5f + 0.5f) * target_height_flt;
					clipped_tris_result[1].screen_pos.z = (p1_projview.z / p1_projview.w);

					clipped_tris_result[2].screen_pos.x = ((p2_projview.x / p2_projview.w) * 0.5f + 0.5f) * target_width_flt;
					clipped_tris_result[2].screen_pos.y = ((p2_projview.y / p2_projview.w) * 0.5f + 0.5f) * target_height_flt;
					clipped_tris_result[2].screen_pos.z = (p2_projview.z / p2_projview.w);

					const float cross_z = (clipped_tris_result[1].screen_pos.x - clipped_tris_result[0].screen_pos.x) * (clipped_tris_result[2].screen_pos.y - clipped_tris_result[0].screen_pos.y) - (clipped_tris_result[2].screen_pos.x - clipped_tris_result[0].screen_pos.x) * (clipped_tris_result[1].screen_pos.y - clipped_tris_result[0].screen_pos.y);
					const bool backface_culling = cross_z > 0.0f;

					if(backface_culling)
					{
						draw_triangle<draw_ctx_t, triangle_t>(tri, clipped_tris_result, draw_hline_function, frame_width, frame_height);
					}
				}
			}
		}
	}

	template<draw_horizontal_line_ctx draw_ctx_t, triangle_list triangle_list_t>
	constexpr void render(const triangle_list_t& triangles, const camera auto& camera, draw_horizontal_line_function<draw_ctx_t, triangle_from_list_t<triangle_list_t>> auto&& draw_hline_function, unsigned_integral auto frame_width, unsigned_integral auto frame_height)
	{
		using triangle_t 				= triangle_from_list_t<triangle_list_t>;
		using vertex_t					= vertex_from_tri_t<triangle_t>;
		using user_defined_iterators_t 	= std::conditional_t<detail::has_user_defined_iterators<draw_ctx_t>, std::decay_t<decltype(std::declval<draw_ctx_t>().begin)>, std::nullptr_t>;
		if constexpr(requires{std::declval<draw_ctx_t>().begin;})
		{
			static_assert(detail::can_assign_vertex<vertex_t, user_defined_iterators_t>, "'begin' member found in 'draw_ctx_t' cannot assign a tri to it");
			static_assert(detail::has_user_defined_iterators<draw_ctx_t>, 				"'begin' member found in 'draw_ctx_t' but does not satisfy the 'has_user_defined_iterators' conditions");
		}

		detail::render_rasterize<draw_ctx_t>(triangles, camera, draw_hline_function, frame_width, frame_height);
	}

	template<triangle_list triangle_list_t>
	constexpr void render_nonshaded(const triangle_list_t& triangles, const camera auto& camera, draw_horizontal_line_function<draw_hline_ctx, triangle_from_list_t<triangle_list_t>> auto&& draw_hline_function, unsigned_integral auto frame_width, unsigned_integral auto frame_height)
	{
		render<draw_hline_ctx, triangle_list_t>(triangles, camera, draw_hline_function, frame_width, frame_height);
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
