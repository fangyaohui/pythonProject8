import pygame
pygame.mixer.init()
screen=pygame.display.set_mode((700,500))
pygame.display.set_caption("木鱼功德")
img1=pygame.image.load("images/muyuluck1.jpg")
img2=pygame.image.load("images/muyulucky2.png")
rect1=img1.get_rect()
rect2=img2.get_rect()
muyulucky = pygame.mixer.Sound('sound/muyu.WAV')
muyulucky.set_volume(0.4)
if pygame.mouse.get_focused():
            # 获取光标位置,2个值
            ball_x, ball_y = pygame.mouse.get_pos()
screen.blit(img1, (-150, -100))
while True:
    for event in pygame.event.get():
        if pygame.Rect.collidepoint(rect2, (ball_x, ball_y)) and event.type==pygame.MOUSEBUTTONDOWN:
            screen.blit(img2, (-150, -100))
            muyulucky.play()
            pygame.display.flip()
        if pygame.Rect.collidepoint(rect1, (ball_x, ball_y)) and event.type==pygame.MOUSEBUTTONUP:
            screen.blit(img1, (-150, -100))
            pygame.display.flip(),
        if event.type==pygame.QUIT:
            pygame.quit()
    pygame.display.flip()