import cv2
import mediapipe as mp
import pygame
import sys

# Initialize Pygame
pygame.init()
screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Hand Gesture Controlled Game")
clock = pygame.time.Clock()

# MediaPipe initialization
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Player class
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super(Player, self).__init__()
        self.surf = pygame.Surface((50, 50))
        self.surf.fill((255, 255, 255))
        self.rect = self.surf.get_rect(center=(screen_width // 2, screen_height - 50))
        self.speed = 5
        self.is_jumping = False
        self.jump_velocity = 10
        self.gravity = 0.5

    def move_left(self):
        if self.rect.left > 0:  # Ensure the player doesn't go out of the screen
            self.rect.move_ip(-self.speed, 0)
    
    def move_right(self):
        if self.rect.right < screen_width:  # Ensure the player doesn't go out of the screen
            self.rect.move_ip(self.speed, 0)

    def jump(self):
        if not self.is_jumping:
            self.is_jumping = True
            self.jump_velocity = -10

    def update(self):
        if self.is_jumping:
            self.rect.move_ip(0, self.jump_velocity)
            self.jump_velocity += self.gravity
            if self.rect.bottom >= screen_height:
                self.rect.bottom = screen_height
                self.is_jumping = False

# Check if the hand is fully open (all fingers extended)
def is_hand_open(hand_landmarks):
    finger_tips_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
    finger_mcp_ids = [3, 5, 9, 13, 17]    # MCP joints

    open_fingers = 0
    for tip_id, mcp_id in zip(finger_tips_ids, finger_mcp_ids):
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[mcp_id].y:  # Tip above MCP
            open_fingers += 1

    return open_fingers == 5  # All fingers open

# Check hand gestures
def check_hand_gesture(hand_landmarks, player):
    # Thumb: Landmark index 4
    thumb_tip_x = hand_landmarks.landmark[4].x
    thumb_ip_x = hand_landmarks.landmark[3].x

    # Pinky finger: Landmark index 20
    pinky_tip_x = hand_landmarks.landmark[20].x
    pinky_ip_x = hand_landmarks.landmark[19].x

    # Detect if hand is open
    if is_hand_open(hand_landmarks):
        player.jump()  # Jump without moving
    else:
        # Detect left move (pinky extended)
        if pinky_tip_x < pinky_ip_x:
            player.move_left()
        # Detect right move (thumb extended)
        elif thumb_tip_x > thumb_ip_x:
            player.move_right()

# Main game loop
def run_game():
    cap = cv2.VideoCapture(0)
    player = Player()
    all_sprites = pygame.sprite.Group()
    all_sprites.add(player)

    with mp_hands.Hands(max_num_hands=1) as hands:
        running = True
        try:
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                # Read image from camera
                ret, frame = cap.read()
                if not ret:
                    break

                # Flip frame for correct hand orientation
                frame = cv2.flip(frame, 1)

                # Convert image to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                # Draw landmarks and detect gestures
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        # Check hand gestures for controlling the player
                        check_hand_gesture(hand_landmarks, player)

                # Display frame with hand landmarks
                cv2.imshow('MediaPipe Hands', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Update the game
                all_sprites.update()

                # Clear screen and redraw player
                screen.fill((0, 0, 0))
                for entity in all_sprites:
                    screen.blit(entity.surf, entity.rect)

                pygame.display.flip()
                clock.tick(30)

        except KeyboardInterrupt:
            print("Game interrupted by user. Exiting...")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            pygame.quit()
            sys.exit()

# Run the game
if __name__ == "__main__":
    run_game()
