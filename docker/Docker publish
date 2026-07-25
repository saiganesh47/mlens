# .github/workflows/docker-publish.yml
name: Docker Publish

on:
  push:
    branches: [main]
    tags:     ["v*.*.*"]
  pull_request:
    branches: [main]

env:
  REGISTRY:   ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:

  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:

      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Debug — show repo structure
        run: |
          echo "=== Root ==="
          ls -la
          echo "=== docker/ ==="
          ls -la docker/ || echo "No docker/ folder"
          echo "=== mlens/ ==="
          ls -la mlens/ || echo "No mlens/ folder"
          echo "=== api/ ==="
          ls -la api/ || echo "No api/ folder"

      - name: Check required files
        id: check
        run: |
          API=false
          DASH=false
          if [ -f "docker/Dockerfile" ] && [ -d "mlens" ] && [ -d "api" ]; then
            API=true
          fi
          if [ -f "docker/Dockerfile.dashboard" ] && [ -d "dashboard" ]; then
            DASH=true
          fi
          echo "api_ready=$API"   >> $GITHUB_OUTPUT
          echo "dash_ready=$DASH" >> $GITHUB_OUTPUT
          echo "API ready: $API"
          echo "Dashboard ready: $DASH"

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GitHub Container Registry
        if: github.event_name != 'pull_request'
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract Docker metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}

      - name: Build and push API image
        if: steps.check.outputs.api_ready == 'true'
        uses: docker/build-push-action@v6
        with:
          context:    .
          file:       ./docker/Dockerfile
          push:       ${{ github.event_name != 'pull_request' }}
          tags:       ${{ steps.meta.outputs.tags }}
          labels:     ${{ steps.meta.outputs.labels }}
          no-cache:   true

      - name: Build and push Dashboard image
        if: steps.check.outputs.dash_ready == 'true'
        uses: docker/build-push-action@v6
        with:
          context:    .
          file:       ./docker/Dockerfile.dashboard
          push:       ${{ github.event_name != 'pull_request' }}
          tags:       ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}-dashboard:latest
          no-cache:   true

      - name: Skip notice
        if: steps.check.outputs.api_ready == 'false'
        run: |
          echo "⚠️ Skipping Docker build — required folders missing."
          echo "Ensure docker/, mlens/, and api/ exist in the repo root."

      - name: Summary
        run: |
          echo "### Docker Publish Summary" >> $GITHUB_STEP_SUMMARY
          echo "| Service | Status |" >> $GITHUB_STEP_SUMMARY
          echo "|---|---|" >> $GITHUB_STEP_SUMMARY
          echo "| API | ${{ steps.check.outputs.api_ready == 'true' && '✅ Built' || '⚠️ Skipped' }} |" >> $GITHUB_STEP_SUMMARY
          echo "| Dashboard | ${{ steps.check.outputs.dash_ready == 'true' && '✅ Built' || '⚠️ Skipped' }} |" >> $GITHUB_STEP_SUMMARY
