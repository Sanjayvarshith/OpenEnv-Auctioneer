$env:LOCAL_IMAGE_NAME="openenv-auctioneer"
$env:API_BASE_URL="https://openrouter.ai/api/v1"
$env:API_KEY="key"
$env:NO_DOCKER="1"

# Set the task you want to test (optional, defaults to "all")
# $env:AUCTIONEER_TASK="easy_headline"

Write-Host "Running inference.py with local OpenRouter setup..."
python inference.py
