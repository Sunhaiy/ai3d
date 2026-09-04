param(
    [string]$MeshDir = "",
    [int]$Samples = 4000,
    [int]$Epochs = 50,
    [int]$BatchSize = 64,
    [int]$Count = 8
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($MeshDir)) {
    python generate_data.py --samples $Samples --resolution 16 --output data/shapes.npz
    if ($LASTEXITCODE -ne 0) { throw "Synthetic dataset generation failed." }
}
else {
    python prepare_meshes.py --input $MeshDir --resolution 16 --output data/shapes.npz --preview-dir data/previews
    if ($LASTEXITCODE -ne 0) { throw "Mesh dataset preparation failed." }
}

python train.py --data data/shapes.npz --epochs $Epochs --batch-size $BatchSize --output runs/tiny3d.pt
if ($LASTEXITCODE -ne 0) { throw "Training failed." }

python sample.py --checkpoint runs/tiny3d.pt --count $Count --output samples
if ($LASTEXITCODE -ne 0) { throw "Sampling failed." }

Write-Host "Done. Import the OBJ files under samples/ into Blender."
