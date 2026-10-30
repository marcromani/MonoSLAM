#if defined(MONOSLAM_SANITIZERS_ENABLED)

#    include <sanitizer/lsan_interface.h>

extern "C" __attribute__((used)) const char*
__lsan_default_suppressions()
{
    return "leak:GuiReceiver::GuiReceiver\n";
}

#endif
