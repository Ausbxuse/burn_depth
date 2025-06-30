{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  nixConfig = {
    substituters = ["https://ausbxuse.cachix.org"];
    trusted-public-keys = ["ausbxuse.cachix.org-1:drebIdu/fI7t62gHgysbbf3C0/tBFgx0f31ID82AmIg="];
  };
  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        opencvGtk-py = pkgs.python312Packages.opencv4.override (old: {enableGtk3 = true;});
        pkgs = import nixpkgs {
          inherit system;
        };
        buildInputs = with pkgs; [
          # stdenv.cc.cc

          #xorg.libX1Z
          SDL2
          SDL2_image
          SDL2_mixer
          SDL2_ttf
          alsa-lib
          boost
          cmake
          cmake
          eigen
          ghc_filesystem
          glib
          libGL
          libjpeg_turbo
          libpulseaudio
          mkcert
          nodejs
          opencv
          openssl
          pcl
          vulkan-headers
          vulkan-loader
          vulkan-tools
          xorg.libX11
          xorg.libXcomposite
          xorg.libXcursor
          xorg.libXdamage
          xorg.libXext
          xorg.libXfixes
          xorg.libXi
          xorg.libXinerama
          xorg.libXrandr
          xorg.libXrender
          xorg.libXxf86vm
          zlib
        ];
      in {
        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.micromamba
            # pkgs.cmake
            # pkgs.pcl
            pkgs.basedpyright
            # pkgs.cyclonedds
            # pkgs.openssl

            pkgs.ffmpeg
            # pkgs.python312Packages.matplotlib
            # pkgs.python312Packages.scipy
            # pkgs.python312Packages.opencv-python
            # opencvGtk-py
          ];
          shellHook = ''
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath buildInputs}:$LD_LIBRARY_PATH
          '';
        };
      }
    );
}
