# Window layout, colors, and paths for the pygame UI.
# Board geometry is just square count times pixel size so stuff lines up.
from pathlib import Path

# How big the window is: 8 squares per side, each square 96px in the pygame
# window.
boardSize = 8
squareSizePixels = 96
windowWidthPixels = boardSize * squareSizePixels
windowHeightPixels = boardSize * squareSizePixels + 56
framesPerSecond = 60
moveAnimationDurationMilliseconds = 220

# Colors for the dark overlays when you get checkmated or when it is only check
# (not mate yet).
checkmateOverlayBackground = (0, 0, 0, 160)
checkmateTitleColor = (255, 230, 120)
checkmateSubtitleColor = (240, 240, 240)

# Full-board check banner (only over the chess grid, same footprint as checkmate
# overlay).
checkOverlayBackground = (70, 15, 25, 185)
checkOverlayTitleColor = (255, 215, 90)
checkOverlaySubtitleColor = (248, 240, 240)
checkBannerDurationMilliseconds = 1800

# Checkerboard colors plus the blue tint for selected square and dots for legal
# moves.
lightSquareColor = (240, 217, 181)
darkSquareColor = (181, 136, 99)
selectionHighlightColor = (80, 160, 255, 120)
legalMoveDotColor = (30, 30, 30, 180)
captureRingColor = (200, 40, 40, 180)
statusBarBackground = (32, 32, 32)
statusBarTextColor = (245, 245, 245)

# Menu screen behind the difficulty buttons before you start a game.
menuBackgroundColor = (28, 32, 40)
menuTitleColor = (255, 235, 180)
menuHintColor = (200, 205, 215)
menuButtonFillColor = (70, 110, 170)
menuButtonBorderColor = (130, 170, 230)
menuButtonHoverColor = (90, 130, 200)
menuButtonTextColor = (255, 255, 255)

resetButtonFillColor = (55, 85, 120)
resetButtonHoverColor = (90, 130, 200)
resetButtonBorderColor = (130, 170, 230)

projectRootDirectory = Path(__file__).resolve().parent.parent
piecesAssetsDirectory = projectRootDirectory / "pieces-png"
