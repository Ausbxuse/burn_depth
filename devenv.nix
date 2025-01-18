{
  pkgs,
  lib,
  ...
}: let
  buildInputs = with pkgs; [
    stdenv.cc.cc
    libuv
    openssl
    zlib
    libGL
    glib
    cmake
    mkcert
    vulkan-headers
    vulkan-loader
    vulkan-tools
    xorg.libX11
  ];
  opencvGtk-py = pkgs.python312Packages.opencv4.override (old: {enableGtk3 = true;});
in {
  cachix.enable = true;
  cachix.pull = ["ausbxuse"];
  cachix.push = "ausbxuse";
  env = {
    LD_LIBRARY_PATH = "${with pkgs; lib.makeLibraryPath buildInputs}";
    CACHIX_AUTH_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJqdGkiOiIyMjY1YTI1ZS1lMDYzLTQ4MGQtYTc4OC0xZTY2NTg1Y2NmYWYiLCJzY29wZXMiOiJ0eCJ9.C5SuwjuEvdOk1QDpa-bjb0OU2sbfuYllrYc_L8Enirg";
    # CMAKE_PREFIX_PATH = "${pkgs.cyclonedds}:${pkgs.cmake}";
  };
  languages.python = {
    enable = true;
    uv = {
      enable = true;
      sync.enable = false;
    };
    venv.enable = true;
    venv.requirements = ./requirements.txt;
  };

  # https://devenv.sh/basics/
  env.GREET = "devenv";

  packages = [
    pkgs.ffmpeg
    pkgs.python312Packages.matplotlib
    opencvGtk-py
  ];

  scripts.hello.exec = ''
    echo hello from $GREET
  '';

  # enterShell = ''
  #   export QT_QPA_PLATFORM_PLUGIN_PATH="${pkgs.libsForQt5.qt5.qtbase.bin}/lib/qt-${pkgs.libsForQt5.qt5.qtbase.version}/plugins"
  # '';

  enterTest = ''
    echo "Running tests"
    git --version | grep --color=auto "${pkgs.git.version}"
  '';
}
