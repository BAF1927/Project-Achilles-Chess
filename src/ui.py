# Pygame front end: difficulty menu, board rendering, input, and hook-ins to ChessGame.
from __future__ import annotations

import chess
import pygame

from .config import (
    boardSize,
    captureRingColor,
    checkBannerDurationMilliseconds,
    checkOverlayBackground,
    checkOverlaySubtitleColor,
    checkOverlayTitleColor,
    checkmateOverlayBackground,
    checkmateSubtitleColor,
    checkmateTitleColor,
    darkSquareColor,
    framesPerSecond,
    legalMoveDotColor,
    lightSquareColor,
    menuBackgroundColor,
    menuButtonBorderColor,
    menuButtonFillColor,
    menuButtonHoverColor,
    menuButtonTextColor,
    menuHintColor,
    menuTitleColor,
    resetButtonBorderColor,
    resetButtonFillColor,
    resetButtonHoverColor,
    selectionHighlightColor,
    squareSizePixels,
    statusBarBackground,
    statusBarTextColor,
    windowHeightPixels,
    windowWidthPixels,
)
from .game import ChessGame, describeCheckmateAttackingPieces


def smoothStepInterpolation(progressUnitInterval: float) -> float:
    # Hermite smoothstep so animated pieces ease in and out instead of moving
    # linearly.
    clampedProgress = max(0.0, min(1.0, progressUnitInterval))
    return clampedProgress * clampedProgress * (3.0 - 2.0 * clampedProgress)


class ChessApplication:
    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((windowWidthPixels, windowHeightPixels))
        pygame.display.set_caption("PyReason Chess")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 22)
        self.titleFont = pygame.font.SysFont("Arial", 40, bold=True)
        self.subFont = pygame.font.SysFont("Arial", 24)
        self.buttonFont = pygame.font.SysFont("Arial", 22, bold=True)
        self.hintFont = pygame.font.SysFont("Arial", 18)
        self.checkBannerFont = pygame.font.SysFont("Arial", 84, bold=True)
        self.applicationRunning = True
        self.applicationPhase = "menu"
        self.selectedDifficultyKey = "medium"
        self.game: ChessGame | None = None
        statusBarTopCoordinate = boardSize * squareSizePixels
        resetButtonWidthPixels, resetButtonHeightPixels = 140, 34
        self.resetButtonRectangle = pygame.Rect(
            windowWidthPixels - resetButtonWidthPixels - 14,
            statusBarTopCoordinate + (56 - resetButtonHeightPixels) // 2,
            resetButtonWidthPixels,
            resetButtonHeightPixels,
        )
        self.difficultyButtonRectangles: dict[str, pygame.Rect] = {}
        self.layoutDifficultyButtons()

        self.checkBannerMessageText: str = ""
        self.checkBannerStartTimeMilliseconds: int = 0
        self.checkBannerLastBoardFingerprint: str = ""
        self.checkBannerLastFrameWasCheckState: bool = False

    def layoutDifficultyButtons(self) -> None:
        centerXPixels = windowWidthPixels // 2
        firstButtonTopPixels = 240
        buttonWidthPixels, buttonHeightPixels = 380, 50
        verticalGapPixels = 12
        self.difficultyButtonRectangles = {}
        for buttonRowIndex, difficultyKey in enumerate(["easy", "medium", "hard", "achilles"]):
            topLeftY = firstButtonTopPixels + buttonRowIndex * (buttonHeightPixels + verticalGapPixels)
            self.difficultyButtonRectangles[difficultyKey] = pygame.Rect(
                centerXPixels - buttonWidthPixels // 2,
                topLeftY,
                buttonWidthPixels,
                buttonHeightPixels,
            )

    def run(self) -> None:
        while self.applicationRunning:
            currentTimeMilliseconds = pygame.time.get_ticks()
            self.handleEvents(currentTimeMilliseconds)
            if self.applicationPhase == "play" and self.game is not None:
                self.game.updateAnimation(currentTimeMilliseconds)
                self.game.moveOnlyWhenTurn(currentTimeMilliseconds)
                self.updateCheckBannerToast(currentTimeMilliseconds)
            self.drawFrame(currentTimeMilliseconds)
            pygame.display.flip()
            self.clock.tick(framesPerSecond)
        pygame.quit()

    def handleEvents(self, currentTimeMilliseconds: int) -> None:
        for pygameEvent in pygame.event.get():
            if pygameEvent.type == pygame.QUIT:
                self.applicationRunning = False
                continue
            if pygameEvent.type == pygame.KEYDOWN and pygameEvent.key == pygame.K_ESCAPE:
                self.applicationRunning = False
                continue
            if self.applicationPhase == "menu" and pygameEvent.type == pygame.MOUSEBUTTONDOWN and pygameEvent.button == 1:
                self.handleMenuClick(pygameEvent.pos)
                continue
            if self.applicationPhase == "play" and self.game is not None:
                if pygameEvent.type == pygame.KEYDOWN:
                    if pygameEvent.key == pygame.K_r:
                        if self.game.state.board.is_game_over():
                            self.applicationPhase = "menu"
                            self.game = None
                            self.checkBannerMessageText = ""
                            self.checkBannerStartTimeMilliseconds = 0
                            self.checkBannerLastBoardFingerprint = ""
                            self.checkBannerLastFrameWasCheckState = False
                        else:
                            self.game.resetBoardToNewGame()
                    continue
                if pygameEvent.type == pygame.MOUSEBUTTONDOWN and pygameEvent.button == 1:
                    clickXPixels, clickYPixels = pygameEvent.pos
                    boardPixelBottom = boardSize * squareSizePixels
                    if clickYPixels >= boardPixelBottom:
                        if self.resetButtonRectangle.collidepoint((clickXPixels, clickYPixels)):
                            if self.game.state.board.is_game_over():
                                self.applicationPhase = "menu"
                                self.game = None
                                self.checkBannerMessageText = ""
                                self.checkBannerStartTimeMilliseconds = 0
                                self.checkBannerLastBoardFingerprint = ""
                                self.checkBannerLastFrameWasCheckState = False
                            else:
                                self.game.resetBoardToNewGame()
                        continue
                    if clickYPixels < boardPixelBottom:
                        self.game.handlePlayerClick(
                            clickXPixels // squareSizePixels,
                            clickYPixels // squareSizePixels,
                            currentTimeMilliseconds,
                        )
                if pygameEvent.type == pygame.FINGERDOWN:
                    fingerXPixels = int(pygameEvent.x * windowWidthPixels)
                    fingerYPixels = int(pygameEvent.y * windowHeightPixels)
                    boardPixelBottom = boardSize * squareSizePixels
                    if fingerYPixels >= boardPixelBottom:
                        if self.resetButtonRectangle.collidepoint((fingerXPixels, fingerYPixels)):
                            if self.game.state.board.is_game_over():
                                self.applicationPhase = "menu"
                                self.game = None
                                self.checkBannerMessageText = ""
                                self.checkBannerStartTimeMilliseconds = 0
                                self.checkBannerLastBoardFingerprint = ""
                                self.checkBannerLastFrameWasCheckState = False
                            else:
                                self.game.resetBoardToNewGame()
                        continue
                    if fingerYPixels < boardPixelBottom and self.game is not None:
                        self.game.handlePlayerClick(
                            fingerXPixels // squareSizePixels,
                            fingerYPixels // squareSizePixels,
                            currentTimeMilliseconds,
                        )

    def handleMenuClick(self, clickPosition: tuple[int, int]) -> None:
        clickPoint = (clickPosition[0], clickPosition[1])
        for difficultyKey, buttonRectangle in self.difficultyButtonRectangles.items():
            if buttonRectangle.collidepoint(clickPoint):
                self.selectedDifficultyKey = difficultyKey
                self.startPlayPhase()
                return

    def startPlayPhase(self) -> None:
        self.applicationPhase = "play"
        self.game = ChessGame(difficulty=self.selectedDifficultyKey)

    def updateCheckBannerToast(self, currentTimeMilliseconds: int) -> None:
        if self.game is None:
            return
        board = self.game.state.board
        if board.is_check() and not board.is_checkmate():
            boardFingerprint = f"{board.fen()}|turn={board.turn}"
            if (
                boardFingerprint != self.checkBannerLastBoardFingerprint
                or not self.checkBannerLastFrameWasCheckState
            ):
                self.checkBannerLastBoardFingerprint = boardFingerprint
                self.checkBannerLastFrameWasCheckState = True
                sideDescription = "White (you)" if board.turn == chess.WHITE else "Black (AI)"
                self.checkBannerMessageText = f"{sideDescription} — king is in check"
                self.checkBannerStartTimeMilliseconds = currentTimeMilliseconds
        else:
            self.checkBannerLastFrameWasCheckState = False

        if (
            self.checkBannerMessageText
            and (currentTimeMilliseconds - self.checkBannerStartTimeMilliseconds) >= checkBannerDurationMilliseconds
        ):
            self.checkBannerMessageText = ""

    def drawFrame(self, currentTimeMilliseconds: int) -> None:
        if self.applicationPhase == "menu":
            self.drawMenuScreen()
        elif self.applicationPhase == "play" and self.game is not None:
            self.drawPlayScreen(currentTimeMilliseconds)

    def drawMenuScreen(self) -> None:
        self.screen.fill(menuBackgroundColor)
        titleSurface = self.titleFont.render("PyReason Chess", True, menuTitleColor)
        self.screen.blit(titleSurface, titleSurface.get_rect(center=(windowWidthPixels // 2, 120)))
        subtitleSurface = self.subFont.render("Developed by Bruno Arriola Flores", True, menuHintColor)
        self.screen.blit(subtitleSurface, subtitleSurface.get_rect(center=(windowWidthPixels // 2, 158)))
        hintLines = [
            "Choose difficulty — straight to the board (PyReason every black move)",
            "First launch slow? Run once:  python -m src.prewarm",
        ]
        hintLineVerticalOffsetPixels = 192
        for hintLine in hintLines:
            hintSurface = self.hintFont.render(hintLine, True, menuHintColor)
            self.screen.blit(hintSurface, hintSurface.get_rect(center=(windowWidthPixels // 2, hintLineVerticalOffsetPixels)))
            hintLineVerticalOffsetPixels += 26
        mousePosition = pygame.mouse.get_pos()
        difficultyLabels = {
            "easy": "Easy",
            "medium": "Medium",
            "hard": "Hard",
            "achilles": "Achilles",
        }
        for difficultyKey, buttonRectangle in self.difficultyButtonRectangles.items():
            mouseIsOverButton = buttonRectangle.collidepoint(mousePosition)
            fillColor = menuButtonHoverColor if mouseIsOverButton else menuButtonFillColor
            pygame.draw.rect(self.screen, fillColor, buttonRectangle, border_radius=12)
            pygame.draw.rect(self.screen, menuButtonBorderColor, buttonRectangle, width=3, border_radius=12)
            labelSurface = self.buttonFont.render(difficultyLabels[difficultyKey], True, menuButtonTextColor)
            self.screen.blit(labelSurface, labelSurface.get_rect(center=buttonRectangle.center))

    def pixelTopLeftOfSquare(self, squareIndex: int) -> tuple[int, int]:
        fileIndex = chess.square_file(squareIndex)
        rankIndexFromTop = boardSize - 1 - chess.square_rank(squareIndex)
        return (fileIndex * squareSizePixels, rankIndexFromTop * squareSizePixels)

    def drawPlayScreen(self, currentTimeMilliseconds: int) -> None:
        assert self.game is not None
        self.drawCheckerboard()
        self.drawSelectionAndLegalMoves()
        self.drawPiecesWithAnimation(currentTimeMilliseconds)
        self.drawCheckBannerOverlay(currentTimeMilliseconds)
        self.drawCheckmateOverlay()
        self.drawStatusBar()

    def drawCheckerboard(self) -> None:
        for boardRankIndex in range(boardSize):
            for boardFileIndex in range(boardSize):
                squareColor = lightSquareColor if (boardRankIndex + boardFileIndex) % 2 == 0 else darkSquareColor
                cellRectangle = pygame.Rect(
                    boardFileIndex * squareSizePixels,
                    boardRankIndex * squareSizePixels,
                    squareSizePixels,
                    squareSizePixels,
                )
                pygame.draw.rect(self.screen, squareColor, cellRectangle)

    def drawSelectionAndLegalMoves(self) -> None:
        gameState = self.game.state
        board = gameState.board
        if gameState.selectedSquare is None:
            return
        if board.turn != chess.WHITE:
            return
        fileIndex = chess.square_file(gameState.selectedSquare)
        rankIndexFromTop = boardSize - 1 - chess.square_rank(gameState.selectedSquare)
        selectionOverlay = pygame.Surface((squareSizePixels, squareSizePixels), pygame.SRCALPHA)
        selectionOverlay.fill(selectionHighlightColor)
        self.screen.blit(selectionOverlay, (fileIndex * squareSizePixels, rankIndexFromTop * squareSizePixels))
        for legalMove in gameState.selectedMoves:
            targetFileIndex = chess.square_file(legalMove.to_square)
            targetRankFromTop = boardSize - 1 - chess.square_rank(legalMove.to_square)
            markerCenter = (
                targetFileIndex * squareSizePixels + squareSizePixels // 2,
                targetRankFromTop * squareSizePixels + squareSizePixels // 2,
            )
            targetOccupant = board.piece_at(legalMove.to_square)
            if targetOccupant is None:
                pygame.draw.circle(self.screen, legalMoveDotColor, markerCenter, squareSizePixels // 8)
            else:
                pygame.draw.circle(
                    self.screen, captureRingColor, markerCenter, squareSizePixels // 2 - 8, width=5
                )

    def drawPiecesWithAnimation(self, currentTimeMilliseconds: int) -> None:
        board = self.game.state.board
        interpolationFactor, activeAnimation = self.game.animationProgress(currentTimeMilliseconds)
        destinationSquareToHideWhileAnimating: int | None = None
        if activeAnimation is not None and interpolationFactor < 1.0:
            destinationSquareToHideWhileAnimating = activeAnimation.toSquare
        for squareIndex in chess.SQUARES:
            if destinationSquareToHideWhileAnimating is not None and squareIndex == destinationSquareToHideWhileAnimating:
                continue
            piece = board.piece_at(squareIndex)
            if piece is None:
                continue
            imageKey = (piece.color, piece.piece_type)
            pieceSurface = self.game.pieceImages[imageKey]
            topLeftX, topLeftY = self.pixelTopLeftOfSquare(squareIndex)
            self.screen.blit(pieceSurface, (topLeftX, topLeftY))
        if activeAnimation is not None and interpolationFactor < 1.0:
            imageKey = (activeAnimation.pieceColor, activeAnimation.pieceType)
            pieceSurface = self.game.pieceImages[imageKey]
            startX, startY = self.pixelTopLeftOfSquare(activeAnimation.fromSquare)
            endX, endY = self.pixelTopLeftOfSquare(activeAnimation.toSquare)
            eased = smoothStepInterpolation(interpolationFactor)
            displayXPixels = int(round(startX + (endX - startX) * eased))
            displayYPixels = int(round(startY + (endY - startY) * eased))
            self.screen.blit(pieceSurface, (displayXPixels, displayYPixels))

    def drawCheckmateOverlay(self) -> None:
        board = self.game.state.board
        if not board.is_checkmate():
            return
        overlaySurface = pygame.Surface((windowWidthPixels, boardSize * squareSizePixels), pygame.SRCALPHA)
        overlaySurface.fill(checkmateOverlayBackground)
        self.screen.blit(overlaySurface, (0, 0))
        winnerLine = "Black wins" if board.turn == chess.WHITE else "White wins"
        attackerSummary = describeCheckmateAttackingPieces(board)
        titleSurface = self.titleFont.render("CHECKMATE", True, checkmateTitleColor)
        winnerSurface = self.subFont.render(winnerLine, True, checkmateSubtitleColor)
        attackerSurface = self.subFont.render(f"Attacking piece: {attackerSummary}", True, checkmateSubtitleColor)
        hintSurface = self.font.render("Back to difficulty", True, checkmateSubtitleColor)
        centerXPixels = windowWidthPixels // 2
        verticalCursorPixels = boardSize * squareSizePixels // 2 - 80
        for surface in (titleSurface, winnerSurface, attackerSurface, hintSurface):
            self.screen.blit(surface, surface.get_rect(center=(centerXPixels, verticalCursorPixels)))
            verticalCursorPixels += 44

    def drawCheckBannerOverlay(self, currentTimeMilliseconds: int) -> None:
        if self.game is None or not self.checkBannerMessageText:
            return
        board = self.game.state.board
        if board.is_checkmate():
            return
        elapsedMilliseconds = currentTimeMilliseconds - self.checkBannerStartTimeMilliseconds
        if elapsedMilliseconds < 0 or elapsedMilliseconds >= checkBannerDurationMilliseconds:
            return
        fadeStrength = max(0.0, 1.0 - (elapsedMilliseconds / float(checkBannerDurationMilliseconds)))
        boardPixelHeight = boardSize * squareSizePixels
        overlaySurface = pygame.Surface((windowWidthPixels, boardPixelHeight), pygame.SRCALPHA)
        backgroundAlpha = max(0, min(255, int(checkOverlayBackground[3] * fadeStrength)))
        overlaySurface.fill(
            (checkOverlayBackground[0], checkOverlayBackground[1], checkOverlayBackground[2], backgroundAlpha)
        )
        self.screen.blit(overlaySurface, (0, 0))
        textAlpha = max(0, min(255, int(255 * fadeStrength)))
        titleSurface = self.checkBannerFont.render("CHECK", True, checkOverlayTitleColor)
        subtitleSurface = self.subFont.render(self.checkBannerMessageText, True, checkOverlaySubtitleColor)
        titleSurface.set_alpha(textAlpha)
        subtitleSurface.set_alpha(textAlpha)
        centerXPixels = windowWidthPixels // 2
        centerYPixels = boardPixelHeight // 2
        self.screen.blit(titleSurface, titleSurface.get_rect(center=(centerXPixels, centerYPixels - 36)))
        self.screen.blit(subtitleSurface, subtitleSurface.get_rect(center=(centerXPixels, centerYPixels + 42)))

    def drawStatusBar(self) -> None:
        statusBarTopCoordinate = boardSize * squareSizePixels
        pygame.draw.rect(
            self.screen,
            statusBarBackground,
            pygame.Rect(0, statusBarTopCoordinate, windowWidthPixels, windowHeightPixels - statusBarTopCoordinate),
        )
        statusLine = self.buildStatusBarText()
        self.screen.blit(self.font.render(statusLine, True, statusBarTextColor), (12, statusBarTopCoordinate + 16))

        board = self.game.state.board if self.game is not None else None
        mousePosition = pygame.mouse.get_pos()
        mouseIsOverReset = self.resetButtonRectangle.collidepoint(mousePosition)
        resetFillColor = resetButtonHoverColor if mouseIsOverReset else resetButtonFillColor
        pygame.draw.rect(self.screen, resetFillColor, self.resetButtonRectangle, border_radius=10)
        pygame.draw.rect(self.screen, resetButtonBorderColor, self.resetButtonRectangle, width=2, border_radius=10)
        resetLabelText = "Back" if (board is not None and board.is_game_over()) else "Reset"
        resetLabelSurface = self.buttonFont.render(resetLabelText, True, (255, 255, 255))
        self.screen.blit(resetLabelSurface, resetLabelSurface.get_rect(center=self.resetButtonRectangle.center))

    def buildStatusBarText(self) -> str:
        state = self.game.state
        board = state.board
        if board.is_checkmate():
            loser = "Black (AI)" if board.turn == chess.WHITE else "White (You)"
            return f"Checkmate — {loser} | {describeCheckmateAttackingPieces(board)} | Back to difficulty"
        if board.is_game_over() and not board.is_checkmate():
            if board.is_stalemate():
                return "Draw — stalemate (king not in check, no legal moves). Back to difficulty"
            if board.is_insufficient_material():
                return "Draw — insufficient material. Back to difficulty"
            if board.is_fifty_moves() or board.is_seventyfive_moves():
                return "Draw — 50-/75-move rule. Back to difficulty"
            if board.is_fivefold_repetition():
                return "Draw — fivefold repetition. Back to difficulty"
            return "Draw. Back to difficulty"
        turnDescription = "White (You)" if board.turn == chess.WHITE else "Black (AI)"
        checkSuffix = " CHECK" if board.is_check() else ""
        aiExplanationTail = f" | {state.aiReasons[0]}" if state.aiReasons else ""
        return f"{turnDescription}{checkSuffix} | diff {self.selectedDifficultyKey}{aiExplanationTail}"


# Skip the menu and open the board directly (useful for tests).
def createChessApplicationSkipMenu(difficulty: str = "medium") -> ChessApplication:
    application = ChessApplication()
    application.applicationPhase = "play"
    application.selectedDifficultyKey = difficulty
    application.game = ChessGame(difficulty=difficulty)
    return application
