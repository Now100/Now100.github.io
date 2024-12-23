Jekyll::Hooks.register :posts, :post_write do |post|
    # 포스트 파일 이름 추출
    filename = File.basename(post.path, ".*")
    
    # 이미지 경로 설정
    img_dir = File.join("assets/img", filename)
    FileUtils.mkdir_p(img_dir) unless File.directory?(img_dir)
  end
  