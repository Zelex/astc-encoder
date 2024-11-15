#ifndef _RAD_TM_
#define _RAD_TM_

/*
   Copyright (C) 2009-2024, Epic Games Tools, Inc.
   Telemetry is a registered trademark of RAD Game Tools
   Phone: 425-893-4300
   Sales: sales3@radgametools.com
   Support: telemetry@radgametools.com
   http://www.radgametools.com
   http://www.radgametools.com/telemetry.htm
   http://www.radgametools.com/telemetry/changelog.htm
*/

#define TM_SDK_MAJOR_VERSION 3
#define TM_SDK_MINOR_VERSION 5
#define TM_SDK_REVISION_NUMBER 0
#define TM_SDK_BUILD_NUMBER 174
#define TM_SDK_VERSION "3.5.0.174"
#define TmBuildVersion 3.5.0.174
#define TM_BUILD_YEAR 2024
#define TM_BUILD_MONTH 10
#define TM_BUILD_DAY 16
#define TM_BUILD_SECOND 43625
#define TM_BUILD_DATE "2024.10.16.43625"
#define TmBuildDate 2024.10.16.43625

#define TM_API_MAJOR_VERSION 2022
#define TM_API_MINOR_VERSION 5
#define TM_API_REVISION_NUMBER 23
#define TM_API_BUILD_NUMBER 51766
#define TM_API_VERSION "2022.05.23.51766"


/* Typedefs */
#ifdef _MSC_VER
	typedef char* va_list;
	typedef __int64 tm_int64;
	typedef unsigned __int64 tm_uint64;
	#define TM_ALIGN(__type, __var, __num) __declspec(align(__num)) __type __var
	#define tm_aligned_uint64 __declspec(align(8)) tm_uint64
#else // !_WIN32
	#include <stdarg.h> // for va_list
	typedef long long tm_int64;
	typedef unsigned long long tm_uint64;
	#define TM_ALIGN(__type, __var, __num) __type __attribute__ ((aligned (__num))) __var
	#define tm_aligned_uint64 tm_uint64 __attribute__ ((aligned (8)))
#endif
typedef unsigned short tm_uint16;
typedef unsigned int tm_uint32;
typedef int tm_int32;
typedef tm_uint64 tm_string;
typedef struct tm_tweak_var
{
	union
	{
		tm_uint64 m_panel_id;
		const char* m_panel_name;
	};

	union
	{
		tm_uint64 m_name_id;
		const char* m_name;
	};

	tm_uint32 m_tweak_type;

	union
	{
		char* m_bool;
		char* m_char;
		int* m_int;
		float* m_float;
		double* m_double;
		char* m_string;
		void* m_address;
	};

} tm_tweak_var;
typedef struct tm_zone_id
{
	tm_uint64 m_enter;
	tm_uint32 m_thread_id;
	tm_uint32 m_depth;
} tm_zone_id;
typedef void* (*tm_file_open_callback_type)(const char* filename, void* user_data);
typedef tm_uint32 (*tm_file_write_callback_type)(void* file_handle, void* data, tm_uint32 data_size, void* user_data);
typedef void (*tm_file_close_callback_type)(void* file_handle, void* user_data);
typedef struct tm_gpu_zone
{	
	tm_uint64 name;
	tm_uint64 filename;
	tm_uint32 line;
	tm_uint32 depth;
} tm_gpu_zone;
typedef void (*tm_tweak_callback_type)(tm_tweak_var* tweak_var, void* user_data);

/* Enums */
typedef enum tm_error
{
	TM_OK = 0,							/* No error */
	TMERR_DISABLED = 1,					/* Telemetry has been compiled away with NTELEMETRY */
	TMERR_INVALID_PARAM = 2,			/* Out of range, null pointer, etc. */
	TMERR_NULL_API = 3,					/* The Telemetry API is NULL so all Telemetry calls will no-op. Usually this means the telemetry DLL was not in your programs path. */
	TMERR_OUT_OF_RESOURCES = 4,			/* Typically out of available memory, string space, etc. */
	TMERR_UNINITIALIZED = 5,			/* A Telemetry API was called before tmInitialize */
	TMERR_BAD_HOSTNAME = 6,				/* Could not resolve hostname */
	TMERR_COULD_NOT_CONNECT = 7,		/* Could not connect to the server */
	TMERR_UNKNOWN_NETWORK = 8,			/* Unknown error in the networking system */
	TMERR_ALREADY_SHUTDOWN = 9,			/* tmShutdown called more than once */
	TMERR_ARENA_TOO_SMALL = 10,			/* buffer passed to tmInitialize was too small */
	TMERR_BAD_HANDSHAKE = 11,			/* handshake with server failed (protocol error) */
	TMERR_UNALIGNED = 12,				/* One more more parameters were not aligned correctly */
	TMERR_NETWORK_NOT_INITIALIZED = 13,	/* Network startup functions were not called, e.g. WSAStartup */
	TMERR_BAD_VERSION = 14,				/* You're using an out of date version of the Telemetry libraries */
	TMERR_BAD_TIMER = 15,				/* The provided user timer is too coarse for Telemetry */
	TMERR_ALREADY_OPENED = 16,			/* Telemetry is already connected (tmOpen was already called). Call tmClose to close the existing connection before calling tmOpen. */
	TMERR_ALREADY_INITIALIZED = 17,		/* tmInitialize was already called. tmInitialized only needs to be called once. */
	TMERR_FILE_OPEN_FAILED = 18,		/* Telemetry couldn't open the file. */
	TMERR_INIT_NETWORKING_FAILED = 19,	/* tmOpen was called with TMOF_INIT_NETWORKING, and that initialization failed. */
	TMERR_THREAD_CREATE_FAILED = 20,	/* Creating the Telemetry thread failed. */
	TMERR_WRONG_PLATFORM = 21,			/* Trying to use Telemetry library on the wrong platform */
	TMERR_UNKNOWN = 0xFFFF,				/* Unknown error occurred */
} tm_error;

typedef enum tm_compression
{
	TMCOMPRESS_LZB = 0,				/* LZB compression */
	TMCOMPRESS_LZ4_DEPRECATED = 1,
	TMCOMPRESS_NONE = 2,			/* No compression */
	TMCOMPRESS_UNKNOWN = 0x7FFFFFFF
} tm_compression;

typedef enum tm_zone_flags
{
	TMZF_NONE = 0,							/* Normal zones are drawn without any special color or dimensions */
	TMZF_STALL = 1 << 1,					/* Stall zones are drawn in red */
	TMZF_IDLE = 1 << 2,						/* Idle zones are drawn in grey. */
	TMZF_CONTEXT_SWITCH  = 1 << 3,			/* Context switch zones are drawn narrower */
	TMZF_PLOT_TIME  = 1 << 4,				/* Plot this zones time */
	TMZF_PLOT_TIME_EXPERIMENTAL = TMZF_PLOT_TIME, /* For compatibility with Telemetry 2 */
	TMZF_HAS_ACCUMULATION  = 1 << 5,		/* (Internal: do not use manually) This zone contains accumulations zones. */
	TMZF_ACCUMULATION  = 1 << 6,			/* (Internal: do not use manually) This zone represents an accumulation. */
	TMZF_LOCK_HOLD  = 1 << 7,				/* (Internal: do not use manually) This zone represents a time when a lock is held. */
	TMZF_LOCK_WAIT  = 1 << 8,				/* (Internal: do not use manually) This zone represents a time while waiting on a lock. */
	TMZF_RECURSIVE  = 1 << 9,				/* (Internal: do not use manually) This zone is a recursive call. */
	TMZF_NO_DURATION_LABEL = 1 << 10,		/* Don't include the zone's duration in the zone label */
	TMZF_KERNEL_SAMPLING_ZONE = 1 << 11,	/* (Internal: do not use manually) This zone is from kernel stack sampling. */
} tm_zone_flags;

typedef enum tm_message_flags
{
	TMMF_SEVERITY_LOG = 0x0001,				/* Message is a standard log message */
	TMMF_SEVERITY_WARNING = 0x0002,			/* Message is a warning (drawn in yellow) */
	TMMF_SEVERITY_ERROR = 0x0004,			/* Message is an error (drawn in red) */
	TMMF_SEVERITY_RESERVED = 0x0008,		/* Unused */
	TMMF_SEVERITY_MASK = 0xf,				/* Low 3-bits determine the message type */
	TMMF_ZONE_LABEL = 0x0010,				/* Replace zone label with this text */
	TMMF_ZONE_SUBLABEL = 0x0020,			/* Print secondary zone label with text */
	TMMF_ZONE_SHOW_IN_PARENTS = 0x0040,		/* Show this not only in the zone's tooltip, but also in the tooltips of any parents */
	TMMF_ZONE_RESERVED01 = 0x0080,			/* Reserved */
	TMMF_ZONE_MASK = 0x00F0,				/* Mask for all of the zone values */
	TMMF_ICON_EXCLAMATION = 0x1000,			/* Show exclamation marker in the zone view for this message */
	TMMF_ICON_NOTE = 0x2000,				/* Show note marker in the zone view for this message */
	TMMF_ICON_QUESTION_MARK = 0x3000,		/* Show question mark (?) marker in the zone view for this message */
	TMMF_ICON_WTF = TMMF_ICON_QUESTION_MARK,/* Show question mark (?) marker in the zone view for this message */
	TMMF_ICON_EXTRA00 = 0x4000,				/* These are placeholders for now until we get some nicer icons */
	TMMF_ICON_EXTRA01 = 0x5000,				/* These are placeholders for now until we get some nicer icons */
	TMMF_ICON_EXTRA02 = 0x6000,				/* These are placeholders for now until we get some nicer icons */
	TMMF_ICON_EXTRA03 = 0x7000,				/* These are placeholders for now until we get some nicer icons */
	TMMF_ICON_EXTRA04 = 0x8000,				/* These are placeholders for now until we get some nicer icons */
	TMMF_ICON_MASK = 0xf000,				/* Mask for all of the icon values */
	TMMF_CALLSTACK = 0x10000,				/* Include the callstack in the message */
	TMMF_UNUSED01 = 0x20000,				/* Unused */
	TMMF_ONLY_ZONE = 0x40000,				/* Use message only for enclosing zone, will not show up in log or on timeline */
	TMMF_ONLY_BITMAP = 0x80000				/* Show message only for bitmaps, will not show up in log or on timeline */
} tm_message_flags;

typedef enum tm_tweak_type
{
	TMTT_BOOL = 1,	/* Boolean data type creates a check box tweak widget */
	TMTT_CHAR,		/* Char data type creates an edit box tweak widget that will allow for hotkey-like behavior */
	TMTT_INT,		/* Integer data type creates an edit box tweak widget with a value slider */
	TMTT_FLOAT,		/* Float data type creates an edit box tweak widget with a value slider */
	TMTT_FLOAT2,	/* 2D float data type creates two edit box tweak widgets with a value slider */
	TMTT_FLOAT3,	/* 3D float data type creates three edit box tweak widgets with a value slider */
	TMTT_FLOAT4,	/* 4D float data type creates four edit box tweak widgets with a value slider */
	TMTT_DOUBLE,	/* Double data type creates an edit box tweak widget with a value slider */
	TMTT_DOUBLE2,	/* 2D double data type creates two edit box tweak widgets with a value slider */
	TMTT_DOUBLE3,	/* 3D double data type creates three edit box tweak widgets with a value slider */
	TMTT_DOUBLE4,	/* 4D floating point data type creates four edit box tweak widgets with a value slider */
	TMTT_COLOR,		/* Color data type creates a color picker tweak widget */
	TMTT_STRING		/* String data type creates an edit box tweak widget for editing strings */
} tm_tweak_type;

typedef enum tm_library_type
{
	TM_RELEASE = 0,							/* Load the release version of the Telemetry library */
	TM_DEBUG = 1							/* Load the debug version of the Telemetry library */
} tm_library_type;

typedef enum tm_open_flags
{
	TMOF_INIT_NETWORKING = 1 << 0,				/* Initialize operating system networking layer. Specify this if you do not already call the platform specific network startup functions (e.g. WSAStartup) prior to tmOpen.*/
	TMOF_CAPTURE_CONTEXT_SWITCHES = 1 << 1,		/* Enable capturing of context switches on platforms that support it. */
	TMOF_SAVE_PLOTS_CSV = 1 << 3,				/* Saves all plots to .CSV files when capture is finished. */
	TMOF_SAVE_ZONES_CSV = 1 << 4,				/* Saves all zones to a .CSV file when capture is finished. */
	TMOF_SAVE_TIME_SPANS_CSV = 1 << 5,			/* Saves all time spans to a .CSV file when capture is finished. */
	TMOF_SAVE_MESSAGES_CSV = 1 << 6,			/* Saves all messages to a .CSV file when capture is finished. */
	TMOF_GENERATE_HTML_REPORT = 1 << 7,			/* Generates an HTML report when the capture is finished. */
	TMOF_BREAK_ON_MISMATCHED_ZONE = 1 << 8,		/* Breaks when a zone underflow or overflow is detected. */
	TMOF_SAVE_ALL_CSV = TMOF_SAVE_PLOTS_CSV | TMOF_SAVE_ZONES_CSV | TMOF_SAVE_TIME_SPANS_CSV | TMOF_SAVE_MESSAGES_CSV /* Saves everything to to .CSV files when capture is finished. */
} tm_open_flags;

typedef enum tm_connection_type
{
	TMCT_TCP = 0,								/* Standard network connection over TCP Socket */
	TMCT_IPC = 1,								/* Uses shared memory to communicate with Telemetry process */
	TMCT_FILE = 2,								/* Captures the TCP Socket stream directly to a file */
	TMCT_USER_PROVIDED = 3,						/* Reserved for future use. */
} tm_connection_type;

typedef enum tm_plot_draw_type
{
	TM_PLOT_DRAW_LINE = 0,						/* Draw plot as a line. */
	TM_PLOT_DRAW_POINTS = 1,					/* Draw plot as points. */
	TM_PLOT_DRAW_STEPPED_XY = 2,				/* Draw plot as a stair step moving horizontally and then vertically. */
	TM_PLOT_DRAW_STEPPED_YX = 3,				/* Draw plot as a stair step moving vertically and then horizontally  */
	TM_PLOT_DRAW_COLUMN = 4,					/* Draw plot as a column graph (AKA bar graph). */
	TM_PLOT_DRAW_LOLLIPOP = 5					/* Draw plot as a lollipop graph. */
}tm_plot_draw_type;

typedef enum tm_plot_units
{
	TM_PLOT_UNITS_REAL = 0,					/* Plot value is a real number. */
	TM_PLOT_UNITS_MEMORY = 1,				/* Plot value is a memory value (in bytes) */
	TM_PLOT_UNITS_HEX = 2,					/* Plot value is shows in hexadecimal notation */
	TM_PLOT_UNITS_INTEGER = 3,				/* Plot value is an integer number */
	TM_PLOT_UNITS_PERCENTAGE_COMPUTED = 4,	/* Display as a percentage of the max, i.e. as (value-min)/(max-min), computed by the client */
	TM_PLOT_UNITS_PERCENTAGE_DIRECT = 5,	/* Display as a percentage (i.e. 0.2187 => 21.87%) */
	TM_PLOT_UNITS_TIME = 6,					/* Plot value is in seconds */
	TM_PLOT_UNITS_TIME_MS = 7,				/* Plot value is in milliseconds */
	TM_PLOT_UNITS_TIME_US = 8,				/* Plot value is in microseconds */
	TM_PLOT_UNITS_TIME_CLOCKS = 9,			/* Plot value is in CPU clocks */
	TM_PLOT_UNITS_TIME_INTERVAL = 10,		/* Plot value is in CPU clocks */
} tm_plot_units;

typedef enum tm_bitmap_flags
{
	TMBF_MONO	= 0 << 0,					/* Use one-channel format */
	TMBF_RGB	= 1 << 0,					/* Use RGB pixel format, 3 components per pixel */
	TMBF_YUV	= 2 << 0,					/* Use YUV pixel format (for *Planes functions) */
	TMBF_ALPHA	= 1 << 3,					/* Adds alpha channel to RGB or YUV formats above */
	TMBF_RGBA	= TMBF_RGB | TMBF_ALPHA,	/* Use RGBA pixel format, 4 components per pixel */
	TMBF_YUVA	= TMBF_YUV | TMBF_ALPHA,	/* Use YUV pixel format (for *Planes functions) */

	TMBF_BYTE	= 0 << 4,  /* Each pixel component is 8-bit byte */
	TMBF_UINT16	= 1 << 4,  /* Each pixel component is 16-bit int */
	TMBF_FLOAT	= 2 << 4,  /* Each pixel component is 32-bit float */
	TMBF_HALF   = 3 << 4,  /* Each pixel component is 16-bit float */
	TMBF_UINT8	= TMBF_BYTE,

	TMBF_YUV_420	= 0 << 8,	/* Use half-size chroma planes (for YUV pixel format) */
	TMBF_YUV_444	= 1 << 8,	/* Use full-size chroma planes (for YUV pixel format) */

	TMBF_YUV_BT601	= 0 << 12,	/* Use BT.601 for YUV color conversion */
	TMBF_YUV_BT709	= 1 << 12,	/* Use BT.709 for YUV color conversion */
	TMBF_YUV_BT2020	= 2 << 12,	/* Use BT.2020 for YUV color conversion */

	TMBF_YUV_FULL		= 0 << 16,	/* Use YUV full range - 0..255 for Y, U and V */
	TMBF_YUV_LIMITED	= 1 << 16,	/* Use YUV limited range - 16..235 for Y, 16..240 for U and V */

	TMBF_YUV_JPEG = TMBF_YUV_420 | TMBF_YUV_BT601 | TMBF_YUV_FULL,	/* Convenience enum for specifying yuv420p format, BT.601, full YUV range */

	TMBF_NORMALIZE	= 8 << 16,	/* Normalize min/max value to full value range for UINT8 and UINT16 types */

	TMBF_FORMAT_MASK		= 0xf << 0,		/* (Internal: do not use manually) */
	TMBF_TYPE_MASK			= 0xf << 4,		/* (Internal: do not use manually) */
	TMBF_YUV_MASK			= 0xf << 8,		/* (Internal: do not use manually) */
	TMBF_YUV_CONVERT_MASK	= 0xf << 12,	/* (Internal: do not use manually) */
	TMBF_YVV_RANGE_MASK		= 0x7 << 16,	/* (Internal: do not use manually) */
} tm_bitmap_flags;

#define tmLoadLibrary(...)
#define tmCheckVersion(...) 0
#define tmInitialize(...) 0
#define tmOpen(...) TMERR_DISABLED
#define tmClose(...)
#define tmShutdown(...)
#define tmGetThreadHandle(...) 0
#define tmTick(...)
#define tmRegisterFileCallbacks(...)
#define tmFlush(...)
#define tmString(...) 0
#define tmStaticString(...) 0
#define tmPrintf(...) 0
#define tmPrintfV(...) 0
#define tmCallStack(...) 0
#define tmZoneColor(...)
#define tmZoneWaitingFor(...)
#define tmSetZoneFlag(...)
#define tmSetCaptureMask(...)
#define tmGetCaptureMask(...) 0
#define tmGetSessionName(...)
#define tmSetPlotInfo(...)
#define tmThreadName(...)
#define tmEndThread(...)
#define tmTrackOrder(...)
#define tmTrackColor(...)
#define tmHash(...) 0
#define tmHashString(...) 0
#define tmHash32(...) 0
#define tmSetMaxThreadCount(...)
#define tmGetMaxThreadCount(...) 0
#define tmSetMaxTimeSpanTrackCount(...)
#define tmGetMaxTimeSpanTrackCount(...) 0
#define tmSetSendBufferSize(...)
#define tmGetSendBufferSize(...) 0
#define tmSetSendBufferCount(...)
#define tmGetSendBufferCount(...) 0
#define tmGetMemoryFootprint(...) 0
#define tmRunning(...) 0
#define tmSetCompression(...)
#define tmSetZoneFilterThreshold(...)
#define tmNewTrackID(...) 0
#define tmTrackName(...)
#define tmThreadTrack(...) 0
#define tmSwitchToFiber(...)
#define tmEndFiber(...)
#define tmGetCurrentThreadId(...) 0
#define tmZoneColorSticky(...)
#define tmSwitchToFiberOnTrack(...) 0
#define tmGetZoneDepth(...) 0
#define tmSecondsToTicks(...) 0
#define tmNewTimeSpanTrackID(...) 0
#define tmScreenshot(...)
#define tmTickAtTime(...)
#define tmContextSwitchAtTime(...)
#define tmMemZoneAtTime(...)
#define tmProfileThread(...)
#define tmSetSamplingInterval(...)
#define tmBitmapRGB(...)
#define tmBitmapPlanes(...)
#define tmBitmapStart(...)
#define tmBitmapEnd(...)
#define tmBitmapSendRGB(...)
#define tmBitmapSendPlane(...)
#define tmSetThreadStackSize(...)
#define tmGetThreadStackSize(...) 0
#define tmSetMaxTweakCount(...)
#define tmGetMaxTweakCount(...) 0
#define tmRegisterTweakCallback(...)
#define tmGetZoneID(...) 0
#define tmDependsOn(...)
#define tmAddDependency(...)
#define tmStartWaitForLock(...)
#define tmStartWaitForLockBase(...)
#define tmStartWaitForLock1(...)
#define tmStartWaitForLock2(...)
#define tmStartWaitForLock3(...)
#define tmStartWaitForLock4(...)
#define tmStartWaitForLock5(...)
#define tmStartWaitForLockEx(...)
#define tmStartWaitForLockExBase(...)
#define tmStartWaitForLockEx1(...)
#define tmStartWaitForLockEx2(...)
#define tmStartWaitForLockEx3(...)
#define tmStartWaitForLockEx4(...)
#define tmStartWaitForLockEx5(...)
#define tmStartWaitForLockManual(...)
#define tmStartWaitForLockManualBase(...)
#define tmStartWaitForLockManual1(...)
#define tmStartWaitForLockManual2(...)
#define tmStartWaitForLockManual3(...)
#define tmStartWaitForLockManual4(...)
#define tmStartWaitForLockManual5(...)
#define tmAcquiredLock(...)
#define tmAcquiredLockBase(...)
#define tmAcquiredLock1(...)
#define tmAcquiredLock2(...)
#define tmAcquiredLock3(...)
#define tmAcquiredLock4(...)
#define tmAcquiredLock5(...)
#define tmAcquiredLockEx(...)
#define tmAcquiredLockExBase(...)
#define tmAcquiredLockEx1(...)
#define tmAcquiredLockEx2(...)
#define tmAcquiredLockEx3(...)
#define tmAcquiredLockEx4(...)
#define tmAcquiredLockEx5(...)
#define tmAcquiredLockManual(...)
#define tmAcquiredLockManualBase(...)
#define tmAcquiredLockManual1(...)
#define tmAcquiredLockManual2(...)
#define tmAcquiredLockManual3(...)
#define tmAcquiredLockManual4(...)
#define tmAcquiredLockManual5(...)
#define tmReleasedLock(...)
#define tmReleasedLockBase(...)
#define tmReleasedLock1(...)
#define tmReleasedLock2(...)
#define tmReleasedLock3(...)
#define tmReleasedLock4(...)
#define tmReleasedLock5(...)
#define tmReleasedLockEx(...)
#define tmReleasedLockExBase(...)
#define tmReleasedLockEx1(...)
#define tmReleasedLockEx2(...)
#define tmReleasedLockEx3(...)
#define tmReleasedLockEx4(...)
#define tmReleasedLockEx5(...)
#define tmEndWaitForLock(...)
#define tmEndWaitForLockBase(...)
#define tmEndWaitForLock1(...)
#define tmEndWaitForLock2(...)
#define tmEndWaitForLock3(...)
#define tmEndWaitForLock4(...)
#define tmEndWaitForLock5(...)
#define tmMessage(...)
#define tmMessageBase(...)
#define tmMessage1(...)
#define tmMessage2(...)
#define tmMessage3(...)
#define tmMessage4(...)
#define tmMessage5(...)
#define tmMessageEx(...)
#define tmMessageExBase(...)
#define tmMessageEx1(...)
#define tmMessageEx2(...)
#define tmMessageEx3(...)
#define tmMessageEx4(...)
#define tmMessageEx5(...)
#define tmPlot(...)
#define tmPlotBase(...)
#define tmPlot1(...)
#define tmPlot2(...)
#define tmPlot3(...)
#define tmPlot4(...)
#define tmPlot5(...)
#define tmPlotAt(...)
#define tmPlotAtBase(...)
#define tmPlotAt1(...)
#define tmPlotAt2(...)
#define tmPlotAt3(...)
#define tmPlotAt4(...)
#define tmPlotAt5(...)
#define tmZone(...)
#define tmZoneBase(...)
#define tmZone1(...)
#define tmZone2(...)
#define tmZone3(...)
#define tmZone4(...)
#define tmZone5(...)
#define tmZoneEx(...)
#define tmZoneExBase(...)
#define tmZoneEx1(...)
#define tmZoneEx2(...)
#define tmZoneEx3(...)
#define tmZoneEx4(...)
#define tmZoneEx5(...)
#define tmTimeSpan(...)
#define tmTimeSpanBase(...)
#define tmTimeSpan1(...)
#define tmTimeSpan2(...)
#define tmTimeSpan3(...)
#define tmTimeSpan4(...)
#define tmTimeSpan5(...)
#define tmTimeSpanEx(...)
#define tmTimeSpanExBase(...)
#define tmTimeSpanEx1(...)
#define tmTimeSpanEx2(...)
#define tmTimeSpanEx3(...)
#define tmTimeSpanEx4(...)
#define tmTimeSpanEx5(...)
#define tmFunction(...)
#define tmFunctionBase(...)
#define tmFunction1(...)
#define tmFunction2(...)
#define tmFunction3(...)
#define tmFunction4(...)
#define tmFunction5(...)
#define tmEnterAccumulationZone(...)
#define tmEnterAccumulationZoneBase(...)
#define tmEnterAccumulationZone1(...)
#define tmEnterAccumulationZone2(...)
#define tmEnterAccumulationZone3(...)
#define tmEnterAccumulationZone4(...)
#define tmEnterAccumulationZone5(...)
#define tmLeaveAccumulationZone(...)
#define tmLeaveAccumulationZoneBase(...)
#define tmLeaveAccumulationZone1(...)
#define tmLeaveAccumulationZone2(...)
#define tmLeaveAccumulationZone3(...)
#define tmLeaveAccumulationZone4(...)
#define tmLeaveAccumulationZone5(...)
#define tmEmitAccumulationZone(...)
#define tmEmitAccumulationZoneBase(...)
#define tmEmitAccumulationZone1(...)
#define tmEmitAccumulationZone2(...)
#define tmEmitAccumulationZone3(...)
#define tmEmitAccumulationZone4(...)
#define tmEmitAccumulationZone5(...)
#define tmBeginTimeSpan(...)
#define tmBeginTimeSpanBase(...)
#define tmBeginTimeSpan1(...)
#define tmBeginTimeSpan2(...)
#define tmBeginTimeSpan3(...)
#define tmBeginTimeSpan4(...)
#define tmBeginTimeSpan5(...)
#define tmBeginTimeSpanEx(...)
#define tmBeginTimeSpanExBase(...)
#define tmBeginTimeSpanEx1(...)
#define tmBeginTimeSpanEx2(...)
#define tmBeginTimeSpanEx3(...)
#define tmBeginTimeSpanEx4(...)
#define tmBeginTimeSpanEx5(...)
#define tmBeginColoredTimeSpan(...)
#define tmBeginColoredTimeSpanBase(...)
#define tmBeginColoredTimeSpan1(...)
#define tmBeginColoredTimeSpan2(...)
#define tmBeginColoredTimeSpan3(...)
#define tmBeginColoredTimeSpan4(...)
#define tmBeginColoredTimeSpan5(...)
#define tmBeginColoredTimeSpanEx(...)
#define tmBeginColoredTimeSpanExBase(...)
#define tmBeginColoredTimeSpanEx1(...)
#define tmBeginColoredTimeSpanEx2(...)
#define tmBeginColoredTimeSpanEx3(...)
#define tmBeginColoredTimeSpanEx4(...)
#define tmBeginColoredTimeSpanEx5(...)
#define tmEndTimeSpan(...)
#define tmEndTimeSpanBase(...)
#define tmEndTimeSpan1(...)
#define tmEndTimeSpan2(...)
#define tmEndTimeSpan3(...)
#define tmEndTimeSpan4(...)
#define tmEndTimeSpan5(...)
#define tmEndTimeSpanEx(...)
#define tmEndTimeSpanExBase(...)
#define tmEndTimeSpanEx1(...)
#define tmEndTimeSpanEx2(...)
#define tmEndTimeSpanEx3(...)
#define tmEndTimeSpanEx4(...)
#define tmEndTimeSpanEx5(...)
#define tmSetTimeSpanName(...)
#define tmSetTimeSpanNameBase(...)
#define tmSetTimeSpanName1(...)
#define tmSetTimeSpanName2(...)
#define tmSetTimeSpanName3(...)
#define tmSetTimeSpanName4(...)
#define tmSetTimeSpanName5(...)
#define tmTimeSpanAtTime(...)
#define tmTimeSpanAtTimeBase(...)
#define tmTimeSpanAtTime1(...)
#define tmTimeSpanAtTime2(...)
#define tmTimeSpanAtTime3(...)
#define tmTimeSpanAtTime4(...)
#define tmTimeSpanAtTime5(...)
#define tmEnter(...)
#define tmEnterBase(...)
#define tmEnter1(...)
#define tmEnter2(...)
#define tmEnter3(...)
#define tmEnter4(...)
#define tmEnter5(...)
#define tmEnterEx(...)
#define tmEnterExBase(...)
#define tmEnterEx1(...)
#define tmEnterEx2(...)
#define tmEnterEx3(...)
#define tmEnterEx4(...)
#define tmEnterEx5(...)
#define tmLeave(...)
#define tmLeaveBase(...)
#define tmLeave1(...)
#define tmLeave2(...)
#define tmLeave3(...)
#define tmLeave4(...)
#define tmLeave5(...)
#define tmLeaveEx(...)
#define tmLeaveExBase(...)
#define tmLeaveEx1(...)
#define tmLeaveEx2(...)
#define tmLeaveEx3(...)
#define tmLeaveEx4(...)
#define tmLeaveEx5(...)
#define tmRenameZone(...)
#define tmRenameZoneBase(...)
#define tmRenameZone1(...)
#define tmRenameZone2(...)
#define tmRenameZone3(...)
#define tmRenameZone4(...)
#define tmRenameZone5(...)
#define tmZoneAtTime(...)
#define tmZoneAtTimeBase(...)
#define tmZoneAtTime1(...)
#define tmZoneAtTime2(...)
#define tmZoneAtTime3(...)
#define tmZoneAtTime4(...)
#define tmZoneAtTime5(...)
#define tmAlloc(...)
#define tmAllocBase(...)
#define tmAlloc1(...)
#define tmAlloc2(...)
#define tmAlloc3(...)
#define tmAlloc4(...)
#define tmAlloc5(...)
#define tmAllocEx(...)
#define tmAllocExBase(...)
#define tmAllocEx1(...)
#define tmAllocEx2(...)
#define tmAllocEx3(...)
#define tmAllocEx4(...)
#define tmAllocEx5(...)
#define tmFree(...)
#define tmFreeBase(...)
#define tmFree1(...)
#define tmFree2(...)
#define tmFree3(...)
#define tmFree4(...)
#define tmFree5(...)
#define tmFreeEx(...)
#define tmFreeExBase(...)
#define tmFreeEx1(...)
#define tmFreeEx2(...)
#define tmFreeEx3(...)
#define tmFreeEx4(...)
#define tmFreeEx5(...)
#define tmAllocAtTime(...)
#define tmAllocAtTimeBase(...)
#define tmAllocAtTime1(...)
#define tmAllocAtTime2(...)
#define tmAllocAtTime3(...)
#define tmAllocAtTime4(...)
#define tmAllocAtTime5(...)
#define tmFreeAtTime(...)
#define tmFreeAtTimeBase(...)
#define tmFreeAtTime1(...)
#define tmFreeAtTime2(...)
#define tmFreeAtTime3(...)
#define tmFreeAtTime4(...)
#define tmFreeAtTime5(...)
#define tmMessageAtTime(...)
#define tmMessageAtTimeBase(...)
#define tmMessageAtTime1(...)
#define tmMessageAtTime2(...)
#define tmMessageAtTime3(...)
#define tmMessageAtTime4(...)
#define tmMessageAtTime5(...)

#endif /* _RAD_TM_ */

