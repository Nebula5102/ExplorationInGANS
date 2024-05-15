from models2 import * 
from torchvision.datasets import ImageFolder


def predict(images):
  with torch.no_grad():
    checkpoint = f"checkpoint_{epoch}"
    plt.imshow(images, interpolation="nearest")
    plt.savefig(f"checkpoints/{checkpoint}.png")


if __name__ == "__main__":


    
    torch.manual_seed(42)
    opt = Opt()

    generator = Generator()
    discriminator = Discriminator()

    generator.cuda()
    discriminator.cuda()
    

    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    train_set = ImageFolder('data/', transform=transform)
    
    batch_size = opt.batch_size 
    train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True
        )
    
    # Optimizers
    generator_optimizer = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
    discriminator_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)

    Tensor = torch.cuda.FloatTensor

    # ----------
    #  Training
    # ----------

    batches_done = 0
    for epoch in range(opt.n_epochs):
        print('Epoch ' + str(epoch) + ' training...')
        start = time()
        for i, (imgs, _) in enumerate(train_loader):
            real_imgs = Variable(imgs.type(Tensor))
            # train Discriminator
            discriminator_optimizer.zero_grad()
            # sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
            # generate a batch of images
            fake_imgs = generator(z).detach()
            # Adversarial loss
            discriminator_loss = torch.mean(discriminator(fake_imgs)) - torch.mean(discriminator(real_imgs))
            discriminator_loss.backward()
            discriminator_optimizer.step()
            # clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-opt.clip_value, opt.clip_value)
            # train the generator every n_critic iterations
            if i % opt.n_critic == 0:
                # train Generator
                generator_optimizer.zero_grad()
                # generate a batch of fake images
                critics_fake_imgs = generator(z)
                # Adversarial loss
                generator_loss = -torch.mean(discriminator(critics_fake_imgs))
                generator_loss.backward()
                generator_optimizer.step()
            batches_done += 1
        end = time()
        elapsed = end - start
        real_samples, labels = next(iter(train_loader))
        print('done, took %.1f seconds.' % elapsed)
        print(f"Loss of Discriminator: {discriminator_loss}")
        print(f"Loss of Generator: {generator_loss}")
        grid = torchvision.utils.make_grid(fake_imgs.data.cpu(), nrow=4)
        img = (np.transpose(grid.detach().numpy(), (1, 2 ,0)) * 255).astype(np.uint8)
        if epoch % 50 == 0:
            predict(img)
        if epoch %50==0: 
            discriminator.save_model(f"model_checkpoints/discriminator_{epoch}.pth")
            generator.save_model(f"model_checkpoints/generator_{epoch}.pth")

    discriminator.save_model(f"final/discriminator_{epoch}.pth")
    generator.save_model(f"final/generator_{epoch}.pth")
