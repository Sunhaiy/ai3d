param(
    [Parameter(Mandatory = $true)]
    [string]$MeshDir,
    [string]$ImageDir = "",
    [int]$Epochs = 100,
    [int]$Views = 4
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ImageDir)) {
    python prepare_image3d.py --meshes $MeshDir --views $Views --output data/image3d_pairs.npz --preview-dir data/image3d_previews
}
else {
    python prepare_image3d.py --meshes $MeshDir --images $ImageDir --output data/image3d_pairs.npz --preview-dir data/image3d_previews
}
if ($LASTEXITCODE -ne 0) { throw "Image/3D pair preparation failed." }

python train_image_to_3d.py --data data/image3d_pairs.npz --epochs $Epochs --output runs/image_to_3d.pt
if ($LASTEXITCODE -ne 0) { throw "Image-to-3D training failed." }

python reconstruct.py --checkpoint runs/image_to_3d.pt --image data/image3d_previews/preview_0000_input.png --output result.obj
if ($LASTEXITCODE -ne 0) { throw "Reconstruction failed." }

Write-Host "Done. Open result.obj in Blender."

