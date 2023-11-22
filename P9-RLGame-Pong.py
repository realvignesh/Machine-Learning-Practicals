import pygame
import random

pygame.init()
screen = pygame.display.set_mode((600, 400))
clock = pygame.time.Clock()

# INITIALIZE Q TABLE
q_table = {}
for i in range(-10, 11):
    for j in range(-10, 11):
        for k in range(-10, 11):
            for l in range(-10, 11):
                q_table[(i, j, k, l)] = [random.uniform(0, 1) for _ in range(3)]

# INITIALIZE GAME VARIABLES
ball_x, ball_y = 300, 200
ball_dx, ball_dy = random.choice([-4, 4]), random.choice([-4, 4])
paddle1_y, paddle2_y = 150, 150
score1, score2 = 0, 0


# UPDATE GAME STATE
def update_game_state(action):
    global ball_x, ball_y, ball_dx, ball_dy, paddle1_y, paddle2_y, score1, score2
    if action == 0 and paddle1_y > 0:
        paddle1_y -= 5
    elif action == 2 and paddle1_y < 300:
        paddle1_y += 5
    ball_x += ball_dx
    ball_y += ball_dy
    if ball_y < 0 or ball_y > 390:
        ball_dy *= -1
    if ball_x < 20 and paddle1_y < ball_y < paddle1_y + 100:
        ball_dx *= -1
        score1 += 1
    elif ball_x > 580 and paddle2_y < ball_y < paddle2_y + 100:
        ball_dx *= -1
        score2 += 1
    elif ball_x < 0 or ball_x > 600:
        ball_x, ball_y = 300, 200
        ball_dx, ball_dy = random.choice([-4, 4]), random.choice([-4, 4])
        score1, score2 = 0, 0
    if ball_y < paddle2_y + 50 and paddle2_y > 0:
        paddle2_y -= 5
    elif ball_y > paddle2_y + 50 and paddle2_y < 300:
        paddle2_y += 5


# DRAW GAME OBJECTS
def draw_game_objects():
    screen.fill((0, 0, 0))
    pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(0, paddle1_y, 10, 100))
    pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(590, paddle2_y, 10, 100))
    pygame.draw.circle(screen, (255, 255, 255), (int(ball_x), int(ball_y)), 10)
    pygame.draw.line(screen, (255, 255, 255), (300, 0), (300, 400))
    font = pygame.font.SysFont('', 30)
    score_text = font.render(str(score1) + " - " + str(score2), True, (255, 255, 255))
    screen.blit(score_text, (260, 10))
    pygame.display.flip()


# RUN GAME LOOP
while True:
    # Get New State and Update Q-Value
    new_state = (int(ball_x / 10) - int(paddle1_y / 10), int(ball_y / 10), int(paddle2_y / 10),
                 int(ball_dx / abs(ball_dx)), int(ball_dy / abs(ball_dy)))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    # Get Current State and Q-Values
    state = new_state
    q_values = q_table.get(state)
    if q_values is None:
        # Initialize New State with Random Q-Values
        q_table[state] = [random.uniform(0, 1) for _ in range(3)]

    # Choose Action with Highest Q-Value
    action = q_table[state].index(max(q_table[state]))

    # Update Game State based on Chosen Action
    update_game_state(action)

    # Draw Game Objects
    draw_game_objects()

    # reward = score1 - score2

    # Limit Game to 60 FPS
    clock.tick(60)
