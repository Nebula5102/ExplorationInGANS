{ 
    inputs = { 
        
        nixpkgs.url = "nixpkgs/nixos-23.11"; 
        
        }; 
        outputs = { self, nixpkgs, ... } @ inputs: 
            let 
            system = "x86_64-linux"; 
            pkgs = import inputs.nixpkgs { 
                inherit system; 
                config = { 
                    allowUnfree = true; 
                    cudaSupport = true; 
                }; 
            }; 
        in { devShells.${system} = { 
            torch = pkgs.mkShell 
            { buildInputs = with pkgs; [ 
                (python3.withPackages (ps: with ps; [
                    numpy 
                    matplotlib 
                    tqdm 
                    pip 
                    torch-bin
                    torchvision-bin
                    scikit-learn
                ])) 
                    cudaPackages.cudatoolkit 
                    cudaPackages.cudnn 
                    nodePackages.prettier 
                    nodePackages.pyright 
                ]; 
                shellHook = '' export LD_LIBRARY_PATH=${pkgs.cudaPackages.cudatoolkit}/lib:${pkgs.cudaPackages.cudnn}/lib:$LD_LIBRARY_PATH 
                fish 
                ''; 
            };
        };
    };
}

