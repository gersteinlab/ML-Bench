{
  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    poetry2nix = { url = "github:nix-community/poetry2nix"; inputs.nixpkgs.follows = "nixpkgs"; };
  };
  outputs = { self, nixpkgs, flake-utils, poetry2nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        inherit (poetry2nix.lib.mkPoetry2Nix { inherit pkgs; }) mkPoetryApplication defaultPoetryOverrides;
      in
      {
        packages.default = mkPoetryApplication {
          python = pkgs.python312;
          projectDir = self;
          overrides = defaultPoetryOverrides.extend
            (self: super: {
              huggingface = super.huggingface.overridePythonAttrs
                (old: {
                  buildInputs = (old.buildInputs or [ ]) ++ [ super.setuptools ];
                });
              rank-bm25 = super.rank-bm25.overridePythonAttrs
                (old: {
                  buildInputs = (old.buildInputs or [ ]) ++ [ super.setuptools super.wheel ];
                  nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ super.setuptools super.wheel ];
                  format = "setuptools";
                  preBuild = ''
                    export PYTHONPATH=$PYTHONPATH:$PWD
                  '';
                  postPatch = ''
                    if [ ! -f version.py ]; then
                      echo "def get_version(): return '${old.version}'" > version.py
                    fi
                  '';
                });
            });
        };
        devShells.default = pkgs.mkShell {
          inputsFrom = [ self.packages.${system}.default ];
          packages = [ pkgs.poetry ];
        };
      });
}
