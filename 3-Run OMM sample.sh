#!/bin/sh
set -e

cd "$(dirname "$0")"

executable="_Bin/Release/OMMSample"
if [ ! -x "$executable" ]; then
    executable="_Bin/Debug/OMMSample"
fi
if [ ! -x "$executable" ]; then
    printf '%s\n' 'The project is not compiled!' >&2
    exit 1
fi
printf "Running '%s' with Vulkan...\n" "$executable"

printf '\n%s\n' 'Choose resolution:'
printf '%s\n' '1 - 1080p' '2 - 1440p' '3 - 2160p' '4 - 1080p (ultra wide)' '5 - 1440p (ultra wide)'
while :; do
    printf 'Choose resolution [1-5]: '
    IFS= read -r choice || exit 1
    case "$choice" in
        1) width=1920; height=1080; break ;;
        2) width=2560; height=1440; break ;;
        3) width=3840; height=2160; break ;;
        4) width=2560; height=1080; break ;;
        5) width=3440; height=1440; break ;;
    esac
done

printf '\n%s\n' '1 - off' '2 - on'
while :; do
    printf 'Vsync [1-2]: '
    IFS= read -r choice || exit 1
    case "$choice" in
        1) vsync=0; break ;;
        2) vsync=1; break ;;
    esac
done

printf '\n%s\n' '1 - off' '2 - ultra performance' '3 - performance' '4 - balanced' '5 - quality'
while :; do
    printf 'DLSS mode [1-5]: '
    IFS= read -r choice || exit 1
    case "$choice" in
        1) dlss=-1; break ;;
        2) dlss=0; break ;;
        3) dlss=1; break ;;
        4) dlss=2; break ;;
        5) dlss=3; break ;;
    esac
done

printf '\n%s\n' '1 - Bistro (interior)' '2 - Bistro (exterior)' '3 - Shader balls' '4 - Kitchen'
while :; do
    printf 'Choose scene [1-4]: '
    IFS= read -r choice || exit 1
    case "$choice" in
        1) scene=Bistro/BistroInterior.gltf; break ;;
        2) scene=Bistro/BistroExterior.gltf; break ;;
        3) scene=ShaderBalls/ShaderBalls.gltf; break ;;
        4) scene=Kitchen/Kitchen.gltf; break ;;
    esac
done

# Without alwaysActive the window never appears: NRIFramework renders no frame until focused, but the window is not shown (and so not focused) until a frame is presented
set -- "--width=$width" "--height=$height" --api=VULKAN "--scene=$scene" "--dlssQuality=$dlss" --alwaysActive
if [ "$vsync" = 1 ]; then
    set -- "$@" --vsync
fi
exec "$executable" "$@"
